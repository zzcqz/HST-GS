/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"
#include "hts.cuh"

// ---------------------------------------------------------------------------
// Host-side HTS / profiling configuration (env vars, no CUDA sync).
// ---------------------------------------------------------------------------

static int hstgsHtsMode()
{
	const char* e = std::getenv("HSTGS_DISABLE_HTS");
	if (e && (std::strcmp(e, "1") == 0 || std::strcmp(e, "true") == 0 ||
	          std::strcmp(e, "TRUE") == 0 || std::strcmp(e, "yes") == 0 ||
	          std::strcmp(e, "YES") == 0))
		return 0;
	const char* m = std::getenv("HSTGS_HTS_MODE");
	if (m) {
		if (std::strcmp(m, "off") == 0 || std::strcmp(m, "0") == 0)
			return 0;
	}
	return 1;
}

/** 0=st4_st2 (default), 1=st4_only. Env: HSTGS_HTS_HIER=st4_only|st4_st2 */
static int hstgsHtsHierMode()
{
	const char* h = std::getenv("HSTGS_HTS_HIER");
	if (!h) return HTS_HIER_ST4_ST2;
	if (std::strcmp(h, "st4_only") == 0 || std::strcmp(h, "1") == 0)
		return HTS_HIER_ST4_ONLY;
	return HTS_HIER_ST4_ST2;
}

/** Runtime depth sort bits for HTS path. Env: HSTGS_DEPTH_SORT_BITS={16,20,24,32} */
static int hstgsDepthSortBits(int hts_mode)
{
	if (hts_mode == 0)
		return 32;
	const char* e = std::getenv("HSTGS_DEPTH_SORT_BITS");
	if (!e)
		return 24;
	int bits = std::atoi(e);
	if (bits != 16 && bits != 20 && bits != 24 && bits != 32)
		return 24;
	return bits;
}

static bool hstgsHtsCollectStats()
{
	const char* e = std::getenv("HSTGS_HTS_STATS");
	return e && (e[0] == '1' || std::strcmp(e, "true") == 0 || std::strcmp(e, "TRUE") == 0);
}

struct HtsFrameStats {
	uint64_t rect_tiles = 0;
	uint64_t aabb_tiles = 0;
	uint64_t hts_tiles = 0;
	uint64_t st4_cand = 0;
	uint64_t st4_kept = 0;
	uint64_t st2_cand = 0;
	uint64_t st2_kept = 0;
	uint64_t tie_count = 0;
	int num_rendered = 0;
	float binning_ms = 0.f;
	float radix_sort_ms = 0.f;
	float ranges_ms = 0.f;
	float render_ms = 0.f;
	size_t sort_temp_bytes = 0;
};

static HtsFrameStats g_hts_stats;

void htsStatsReset()
{
	g_hts_stats = HtsFrameStats{};
}

uint64_t htsStatsStKept() { return g_hts_stats.st4_kept; }
uint64_t htsStatsStCand() { return g_hts_stats.st4_cand; }
uint64_t htsStatsRectTiles() { return g_hts_stats.rect_tiles; }
uint64_t htsStatsAabbTiles() { return g_hts_stats.aabb_tiles; }
uint64_t htsStatsHtsTiles() { return g_hts_stats.hts_tiles; }
uint64_t htsStatsSt2Cand() { return g_hts_stats.st2_cand; }
uint64_t htsStatsSt2Kept() { return g_hts_stats.st2_kept; }
uint64_t htsStatsTieCount() { return g_hts_stats.tie_count; }
int htsStatsNumRendered() { return g_hts_stats.num_rendered; }
float htsProfileBinningMs() { return g_hts_stats.binning_ms; }
float htsProfileRadixSortMs() { return g_hts_stats.radix_sort_ms; }
float htsProfileRangesMs() { return g_hts_stats.ranges_ms; }
float htsProfileRenderMs() { return g_hts_stats.render_ms; }
size_t htsProfileSortTempBytes() { return g_hts_stats.sort_temp_bytes; }

static void countDepthTies(const uint64_t* keys, int L, int depth_sort_bits)
{
	if (L < 2) return;
	uint64_t tie_count = 0;
	const uint64_t depth_mask = (depth_sort_bits >= 64)
		? ~0ULL
		: ((1ULL << depth_sort_bits) - 1ULL);
	for (int i = 1; i < L; ++i) {
		uint32_t tile_prev = (uint32_t)(keys[i - 1] >> depth_sort_bits);
		uint32_t tile_curr = (uint32_t)(keys[i] >> depth_sort_bits);
		if (tile_prev == 0xFFFFFFFFu || tile_curr == 0xFFFFFFFFu)
			continue;
		if (tile_prev == tile_curr &&
		    (keys[i - 1] & depth_mask) == (keys[i] & depth_mask))
			tie_count++;
	}
	g_hts_stats.tie_count = tie_count;
}

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
// hts_mode==0: native 3DGS (circular getRect + 32-bit depth keys).
// hts_mode!=0: compact ellipse AABB + hierarchical cull + 24-bit depth keys.
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const float4* conic_opacity,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid,
	int hts_mode,
	int hts_hier_mode,
	int depth_sort_bits,
	int depth_sort_shift,
	uint64_t* d_rect_tiles,
	uint64_t* d_aabb_tiles,
	uint64_t* d_hts_tiles,
	uint64_t* d_st4_cand,
	uint64_t* d_st4_kept,
	uint64_t* d_st2_cand,
	uint64_t* d_st2_kept)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	if (radii[idx] <= 0)
		return;

	uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];

	if (hts_mode == 0) {
		uint2 rect_min, rect_max;
		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);
		const int rect_area = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
		if (d_rect_tiles)
			atomicAdd((unsigned long long*)d_rect_tiles, (unsigned long long)rect_area);
		if (d_hts_tiles)
			atomicAdd((unsigned long long*)d_hts_tiles, (unsigned long long)rect_area);
		for (int y = rect_min.y; y < rect_max.y; y++) {
			for (int x = rect_min.x; x < rect_max.x; x++) {
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= (uint64_t)__float_as_uint(depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
		return;
	}

	const uint32_t off_start = off;
	const uint32_t off_end = offsets[idx];
	const uint64_t invalid_key = ((uint64_t)0xFFFFFFFFu) << depth_sort_bits;

	const float4 con_o = conic_opacity[idx];
	const float alpha_log = logf(255.0f * con_o.w);
	if (alpha_log <= 0.0f) {
		radii[idx] = 0;
		while (off < off_end) {
			gaussian_keys_unsorted[off] = invalid_key;
			gaussian_values_unsorted[off] = 0;
			off++;
		}
		return;
	}
	const float q = 2.0f * alpha_log;
	const float det_conic = con_o.x * con_o.z - con_o.y * con_o.y;
	if (det_conic <= 0.0f) {
		radii[idx] = 0;
		while (off < off_end) {
			gaussian_keys_unsorted[off] = invalid_key;
			gaussian_values_unsorted[off] = 0;
			off++;
		}
		return;
	}
	const float extent_x = sqrtf(fmaxf(0.0f, q * con_o.z / det_conic));
	const float extent_y = sqrtf(fmaxf(0.0f, q * con_o.x / det_conic));

	const int tile_min_x = max(0, (int)floorf((points_xy[idx].x - extent_x) / (float)BLOCK_X));
	const int tile_min_y = max(0, (int)floorf((points_xy[idx].y - extent_y) / (float)BLOCK_Y));
	const int tile_max_x = min((int)grid.x, (int)floorf((points_xy[idx].x + extent_x) / (float)BLOCK_X) + 1);
	const int tile_max_y = min((int)grid.y, (int)floorf((points_xy[idx].y + extent_y) / (float)BLOCK_Y) + 1);
	if (tile_max_x <= tile_min_x || tile_max_y <= tile_min_y) {
		radii[idx] = 0;
		while (off < off_end) {
			gaussian_keys_unsorted[off] = invalid_key;
			gaussian_values_unsorted[off] = 0;
			off++;
		}
		return;
	}

	uint2 rect_min, rect_max;
	getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);
	const int rect_area = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	const int aabb_area = (tile_max_x - tile_min_x) * (tile_max_y - tile_min_y);
	if (d_rect_tiles)
		atomicAdd((unsigned long long*)d_rect_tiles, (unsigned long long)rect_area);
	if (d_aabb_tiles)
		atomicAdd((unsigned long long*)d_aabb_tiles, (unsigned long long)aabb_area);

	const uint32_t depth_lo = __float_as_uint(depths[idx]) >> depth_sort_shift;
	const uint32_t emitted = htsProcessTiles(
		points_xy[idx], con_o.x, con_o.y, con_o.z, q,
		tile_min_x, tile_min_y, tile_max_x, tile_max_y,
		hts_mode,
		hts_hier_mode,
		depth_sort_bits,
		d_st4_cand, d_st4_kept,
		d_st2_cand, d_st2_kept,
		true,
		gaussian_keys_unsorted, gaussian_values_unsorted,
		&off, off_end,
		(uint32_t)idx, depth_lo, (int)grid.x);

	if (d_hts_tiles)
		atomicAdd((unsigned long long*)d_hts_tiles, (unsigned long long)emitted);

	if (off == off_start)
		radii[idx] = 0;
	while (off < off_end) {
		gaussian_keys_unsorted[off] = invalid_key;
		gaussian_values_unsorted[off] = 0;
		off++;
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. depth_sort_bits: 32=vanilla, 24=HTS (with invalid pad).
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges, int depth_sort_bits)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> depth_sort_bits;
	const bool skip_invalid = (depth_sort_bits < 32);
	const bool curr_valid = !skip_invalid || (currtile != 0xFFFFFFFFu);
	if (idx == 0) {
		if (curr_valid) ranges[currtile].x = 0;
	} else {
		uint32_t prevtile = point_list_keys[idx - 1] >> depth_sort_bits;
		const bool prev_valid = !skip_invalid || (prevtile != 0xFFFFFFFFu);
		if (currtile != prevtile) {
			if (prev_valid) ranges[prevtile].y = idx;
			if (curr_valid) ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1 && curr_valid)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	obtain(chunk, img.max_contrib, N, 128);
	obtain(chunk, img.pixel_colors, N * NUM_CHANNELS, 128);
	obtain(chunk, img.bucket_count, N, 128);
	obtain(chunk, img.bucket_offsets, N, 128);
	cub::DeviceScan::InclusiveSum(nullptr, img.bucket_count_scan_size, img.bucket_count, img.bucket_count, N);
	obtain(chunk, img.bucket_count_scanning_space, img.bucket_count_scan_size, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

CudaRasterizer::SampleState CudaRasterizer::SampleState::fromChunk(char*& chunk, size_t C)
{
	SampleState sample;
	// One tile id per bucket (not per pixel). T/ar store per-pixel snapshots: C * BLOCK_SIZE.
	obtain(chunk, sample.bucket_to_tile, C, 128);
	obtain(chunk, sample.T, C * BLOCK_SIZE, 128);
	obtain(chunk, sample.ar, NUM_CHANNELS * C * BLOCK_SIZE, 128);
	return sample;
}

__global__ void perTileBucketCount(int T, uint2* ranges, uint32_t* bucketCount) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= T)
		return;
	uint2 range = ranges[idx];
	int num_splats = range.y - range.x;
	int num_buckets = (num_splats + 31) / 32;
	bucketCount[idx] = (uint32_t)num_buckets;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
std::tuple<int, int> CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	std::function<char* (size_t)> sampleBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const bool collect_stats = hstgsHtsCollectStats();
	const int hts_mode = hstgsHtsMode();
	const int hts_hier_mode = hstgsHtsHierMode();
	const int depth_sort_bits = hstgsDepthSortBits(hts_mode);
	const int depth_sort_shift = 32 - depth_sort_bits;

	cudaEvent_t ev_bin0, ev_bin1, ev_sort0, ev_sort1, ev_rng0, ev_rng1, ev_rnd0, ev_rnd1;
	cudaEventCreate(&ev_bin0); cudaEventCreate(&ev_bin1);
	cudaEventCreate(&ev_sort0); cudaEventCreate(&ev_sort1);
	cudaEventCreate(&ev_rng0); cudaEventCreate(&ev_rng1);
	cudaEventCreate(&ev_rnd0); cudaEventCreate(&ev_rnd1);
	cudaEventRecord(ev_bin0);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		false,
		hts_mode,
		hts_hier_mode
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	uint64_t* d_rect = nullptr;
	uint64_t* d_aabb = nullptr;
	uint64_t* d_hts = nullptr;
	uint64_t* d_st4_cand = nullptr;
	uint64_t* d_st4_kept = nullptr;
	uint64_t* d_st2_cand = nullptr;
	uint64_t* d_st2_kept = nullptr;
	if (collect_stats) {
		CHECK_CUDA(cudaMalloc(&d_rect, sizeof(uint64_t)), debug);
		CHECK_CUDA(cudaMalloc(&d_aabb, sizeof(uint64_t)), debug);
		CHECK_CUDA(cudaMalloc(&d_hts, sizeof(uint64_t)), debug);
		CHECK_CUDA(cudaMalloc(&d_st4_cand, sizeof(uint64_t)), debug);
		CHECK_CUDA(cudaMalloc(&d_st4_kept, sizeof(uint64_t)), debug);
		CHECK_CUDA(cudaMalloc(&d_st2_cand, sizeof(uint64_t)), debug);
		CHECK_CUDA(cudaMalloc(&d_st2_kept, sizeof(uint64_t)), debug);
		CHECK_CUDA(cudaMemset(d_rect, 0, sizeof(uint64_t)), debug);
		CHECK_CUDA(cudaMemset(d_aabb, 0, sizeof(uint64_t)), debug);
		CHECK_CUDA(cudaMemset(d_hts, 0, sizeof(uint64_t)), debug);
		CHECK_CUDA(cudaMemset(d_st4_cand, 0, sizeof(uint64_t)), debug);
		CHECK_CUDA(cudaMemset(d_st4_kept, 0, sizeof(uint64_t)), debug);
		CHECK_CUDA(cudaMemset(d_st2_cand, 0, sizeof(uint64_t)), debug);
		CHECK_CUDA(cudaMemset(d_st2_kept, 0, sizeof(uint64_t)), debug);
	}

	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.conic_opacity,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid,
		hts_mode,
		hts_hier_mode,
		depth_sort_bits,
		depth_sort_shift,
		d_rect,
		d_aabb,
		d_hts,
		d_st4_cand,
		d_st4_kept,
		d_st2_cand,
		d_st2_kept);
	CHECK_CUDA(, debug)
	cudaEventRecord(ev_bin1);

	if (d_rect) {
		uint64_t v = 0;
		CHECK_CUDA(cudaMemcpy(&v, d_rect, sizeof(uint64_t), cudaMemcpyDeviceToHost), debug);
		g_hts_stats.rect_tiles = v;
		cudaFree(d_rect);
	}
	if (d_aabb) {
		uint64_t v = 0;
		CHECK_CUDA(cudaMemcpy(&v, d_aabb, sizeof(uint64_t), cudaMemcpyDeviceToHost), debug);
		g_hts_stats.aabb_tiles = v;
		cudaFree(d_aabb);
	}
	if (d_hts) {
		uint64_t v = 0;
		CHECK_CUDA(cudaMemcpy(&v, d_hts, sizeof(uint64_t), cudaMemcpyDeviceToHost), debug);
		g_hts_stats.hts_tiles = v;
		cudaFree(d_hts);
	}
	if (d_st4_cand) {
		CHECK_CUDA(cudaMemcpy(&g_hts_stats.st4_cand, d_st4_cand, sizeof(uint64_t), cudaMemcpyDeviceToHost), debug);
		CHECK_CUDA(cudaMemcpy(&g_hts_stats.st4_kept, d_st4_kept, sizeof(uint64_t), cudaMemcpyDeviceToHost), debug);
		CHECK_CUDA(cudaMemcpy(&g_hts_stats.st2_cand, d_st2_cand, sizeof(uint64_t), cudaMemcpyDeviceToHost), debug);
		CHECK_CUDA(cudaMemcpy(&g_hts_stats.st2_kept, d_st2_kept, sizeof(uint64_t), cudaMemcpyDeviceToHost), debug);
		cudaFree(d_st4_cand); cudaFree(d_st4_kept);
		cudaFree(d_st2_cand); cudaFree(d_st2_kept);
	}

	g_hts_stats.num_rendered = num_rendered;
	g_hts_stats.sort_temp_bytes = binningState.sorting_size;

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);
	cudaEventRecord(ev_sort0);

	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, depth_sort_bits + bit), debug)

	cudaEventRecord(ev_sort1);

	if (collect_stats && num_rendered > 1) {
		std::vector<uint64_t> host_keys(num_rendered);
		CHECK_CUDA(cudaMemcpy(host_keys.data(), binningState.point_list_keys,
			num_rendered * sizeof(uint64_t), cudaMemcpyDeviceToHost), debug);
		countDepthTies(host_keys.data(), num_rendered, depth_sort_bits);
	}

	int num_tiles = tile_grid.x * tile_grid.y;
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	cudaEventRecord(ev_rng0);

	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges,
			depth_sort_bits);
	CHECK_CUDA(, debug)
	cudaEventRecord(ev_rng1);

	unsigned int bucket_sum = 0;
	SampleState sampleState = {};
	const uint32_t* bucket_offsets_ptr = nullptr;
	uint32_t* bucket_to_tile_ptr = nullptr;
	__half* sampled_T_ptr = nullptr;
	__half* sampled_ar_ptr = nullptr;

	if (hts_mode != 0) {
		// FastGS-style bucket snapshots (HTS-only acceleration for backward).
		perTileBucketCount<<<(num_tiles + 255) / 256, 256>>>(num_tiles, imgState.ranges, imgState.bucket_count);
		CHECK_CUDA(cub::DeviceScan::InclusiveSum(
			imgState.bucket_count_scanning_space,
			imgState.bucket_count_scan_size,
			imgState.bucket_count,
			imgState.bucket_offsets,
			num_tiles), debug)
		CHECK_CUDA(cudaMemcpy(&bucket_sum, imgState.bucket_offsets + num_tiles - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost), debug);
		size_t sample_chunk_size = required<SampleState>(bucket_sum);
		char* sample_chunkptr = sampleBuffer(sample_chunk_size);
		sampleState = SampleState::fromChunk(sample_chunkptr, bucket_sum);
		bucket_offsets_ptr = imgState.bucket_offsets;
		bucket_to_tile_ptr = sampleState.bucket_to_tile;
		sampled_T_ptr = sampleState.T;
		sampled_ar_ptr = sampleState.ar;
	} else {
		// Vanilla path: empty sample buffer so Python autograd still works (B=0).
		char* sample_chunkptr = sampleBuffer(128);
		(void)sample_chunkptr;
	}

	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	cudaEventRecord(ev_rnd0);
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		bucket_offsets_ptr, bucket_to_tile_ptr,
		sampled_T_ptr, sampled_ar_ptr,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		imgState.max_contrib,
		imgState.pixel_colors,
		background,
		out_color,
		radii), debug)
	cudaEventRecord(ev_rnd1);
	cudaEventSynchronize(ev_rnd1);

	float ms_bin = 0.f, ms_sort = 0.f, ms_rng = 0.f, ms_rnd = 0.f;
	cudaEventElapsedTime(&ms_bin, ev_bin0, ev_bin1);
	cudaEventElapsedTime(&ms_sort, ev_sort0, ev_sort1);
	cudaEventElapsedTime(&ms_rng, ev_rng0, ev_rng1);
	cudaEventElapsedTime(&ms_rnd, ev_rnd0, ev_rnd1);
	g_hts_stats.binning_ms = ms_bin;
	g_hts_stats.radix_sort_ms = ms_sort;
	g_hts_stats.ranges_ms = ms_rng;
	g_hts_stats.render_ms = ms_rnd;

	cudaEventDestroy(ev_bin0); cudaEventDestroy(ev_bin1);
	cudaEventDestroy(ev_sort0); cudaEventDestroy(ev_sort1);
	cudaEventDestroy(ev_rng0); cudaEventDestroy(ev_rng1);
	cudaEventDestroy(ev_rnd0); cudaEventDestroy(ev_rnd1);

	return std::make_tuple(num_rendered, (int)bucket_sum);
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R, int B,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	char* sample_buffer,
	const float* dL_dpix,
	const float* dL_invdepths,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dinvdepth,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool antialiasing,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);
	// Bucket sample state only valid when HTS path allocated buckets (B > 0).
	SampleState sampleState = {};
	if (B > 0)
		sampleState = SampleState::fromChunk(sample_buffer, B);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height, R, B,
		imgState.bucket_offsets,
		sampleState.bucket_to_tile,
		sampleState.T,
		sampleState.ar,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		geomState.depths,
		imgState.accum_alpha,
		imgState.n_contrib,
		imgState.max_contrib,
		imgState.pixel_colors,
		dL_dpix,
		dL_invdepths,
		(float4*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dinvdepth), debug);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		opacities,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float4*)dL_dmean2D,
		dL_dconic,
		dL_dinvdepth,
		dL_dopacity,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		antialiasing), debug);
}
