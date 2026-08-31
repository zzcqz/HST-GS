/*
 * Shared Hierarchical Tile Selection (HTS) helpers.
 * Used by preprocess (count) and duplicateWithKeys (emit) when HTS is enabled.
 *
 * When HTS is off (hts_mode==0), callers use native 3DGS paths instead
 * (circular getRect + 32-bit depth keys; this header is not used for emit).
 *
 * Policy (hts_mode != 0):
 *  - small footprint (few AABB tiles): emit/count AABB directly (skip tests)
 *  - large footprint: 4x4 coarse cull -> 2x2 mid cull -> accept all tiles in kept 2x2
 *    (or 4x4-only mode: emit all tiles in kept 4x4 block)
 *  - conservative true min_q (no false cull); depth keys use runtime depth_sort_bits
 */
#pragma once

#include "config.h"
#include <cstdint>
#include <cuda_runtime.h>

constexpr int HTS_ST4 = 4;
constexpr int HTS_ST2 = 2;
/** AABB with this many tiles or fewer skips hierarchy (test cost not worth it). */
constexpr int HTS_SMALL_FOOTPRINT_TILES = 16;

/** Hierarchy ablation modes (when hts_mode != 0). */
constexpr int HTS_HIER_ST4_ST2 = 0;  // default production: 4x4 -> 2x2
constexpr int HTS_HIER_ST4_ONLY = 1; // 4x4 coarse cull only, emit full 4x4 block

__device__ __forceinline__ float htsClampf(float v, float lo, float hi)
{
	return fminf(hi, fmaxf(lo, v));
}

/** Min of q(u,v)=a*u^2+2*b*u*v+c*v^2 over axis-aligned rectangle. */
__device__ __forceinline__ float htsQuadMinRect(
	float a, float b, float c,
	float u0, float u1, float v0, float v1)
{
	if (u0 <= 0.0f && 0.0f <= u1 && v0 <= 0.0f && 0.0f <= v1)
		return 0.0f;

	float best = 1e30f;

	for (int k = 0; k < 2; ++k) {
		const float v = (k == 0) ? v0 : v1;
		const float u_star = htsClampf(-b * v / a, u0, u1);
		best = fminf(best, a * u_star * u_star + 2.0f * b * u_star * v + c * v * v);
	}
	for (int k = 0; k < 2; ++k) {
		const float u = (k == 0) ? u0 : u1;
		const float v_star = htsClampf(-b * u / c, v0, v1);
		best = fminf(best, a * u * u + 2.0f * b * u * v_star + c * v_star * v_star);
	}
	best = fminf(best, a * u0 * u0 + 2.0f * b * u0 * v0 + c * v0 * v0);
	best = fminf(best, a * u0 * u0 + 2.0f * b * u0 * v1 + c * v1 * v1);
	best = fminf(best, a * u1 * u1 + 2.0f * b * u1 * v0 + c * v0 * v0);
	best = fminf(best, a * u1 * u1 + 2.0f * b * u1 * v1 + c * v1 * v1);
	return best;
}

/**
 * Whether a pixel-space rectangle may still produce alpha >= 1/255.
 * Conservative: cull only if true min_q over the rectangle exceeds q_thr.
 */
__device__ __forceinline__ bool htsRectMayContribute(
	float a, float b, float c, float q_thr,
	float u0, float u1, float v0, float v1)
{
	return !(htsQuadMinRect(a, b, c, u0, u1, v0, v1) > q_thr);
}

__device__ __forceinline__ void htsBlockLocalRect(
	int tmin_x, int tmin_y, int tmax_x, int tmax_y, float2 mean,
	float& u0, float& u1, float& v0, float& v1)
{
	u0 = (float)(tmin_x * BLOCK_X) - mean.x;
	u1 = (float)(tmax_x * BLOCK_X - 1) - mean.x;
	v0 = (float)(tmin_y * BLOCK_Y) - mean.y;
	v1 = (float)(tmax_y * BLOCK_Y - 1) - mean.y;
}

__device__ __forceinline__ void htsAcceptTile(
	bool do_emit,
	uint32_t& count,
	uint64_t* keys,
	uint32_t* values,
	uint32_t* inout_off,
	uint32_t off_end,
	uint32_t gauss_idx,
	uint32_t depth_lo,
	int grid_x,
	int tx, int ty,
	int depth_sort_bits)
{
	if (do_emit) {
		uint32_t off = *inout_off;
		if (off >= off_end)
			return;
		uint64_t key = (uint64_t)(ty * grid_x + tx);
		key <<= depth_sort_bits;
		key |= (uint64_t)depth_lo;
		keys[off] = key;
		values[off] = gauss_idx;
		*inout_off = off + 1;
	}
	count++;
}

/**
 * Shared count/emit walk over tiles covered by the ellipse AABB.
 *
 * hts_hier_mode: HTS_HIER_ST4_ST2 (default) or HTS_HIER_ST4_ONLY.
 * depth_sort_bits: runtime depth key width (16/20/24/32).
 */
__device__ inline uint32_t htsProcessTiles(
	float2 mean,
	float a, float b, float c, float q_thr,
	int tile_min_x, int tile_min_y, int tile_max_x, int tile_max_y,
	int hts_mode,
	int hts_hier_mode,
	int depth_sort_bits,
	uint64_t* st4_cand,
	uint64_t* st4_kept,
	uint64_t* st2_cand,
	uint64_t* st2_kept,
	bool do_emit,
	uint64_t* keys,
	uint32_t* values,
	uint32_t* inout_off,
	uint32_t off_end,
	uint32_t gauss_idx,
	uint32_t depth_lo,
	int grid_x)
{
	const int tw = tile_max_x - tile_min_x;
	const int th = tile_max_y - tile_min_y;
	if (tw <= 0 || th <= 0)
		return 0;

	uint32_t count = 0;

	// --- small footprint or HTS off: plain AABB ---
	if (hts_mode == 0 || tw * th <= HTS_SMALL_FOOTPRINT_TILES) {
		for (int ty = tile_min_y; ty < tile_max_y; ++ty)
			for (int tx = tile_min_x; tx < tile_max_x; ++tx)
				htsAcceptTile(do_emit, count, keys, values, inout_off, off_end,
					gauss_idx, depth_lo, grid_x, tx, ty, depth_sort_bits);
		return count;
	}

	// --- large footprint: 4x4 coarse -> (optional) 2x2 mid -> emit ---
	const int st4_min_x = tile_min_x / HTS_ST4;
	const int st4_min_y = tile_min_y / HTS_ST4;
	const int st4_max_x = (tile_max_x + HTS_ST4 - 1) / HTS_ST4;
	const int st4_max_y = (tile_max_y + HTS_ST4 - 1) / HTS_ST4;

	for (int sty4 = st4_min_y; sty4 < st4_max_y; ++sty4) {
		for (int stx4 = st4_min_x; stx4 < st4_max_x; ++stx4) {
			const int b4_min_x = max(tile_min_x, stx4 * HTS_ST4);
			const int b4_max_x = min(tile_max_x, (stx4 + 1) * HTS_ST4);
			const int b4_min_y = max(tile_min_y, sty4 * HTS_ST4);
			const int b4_max_y = min(tile_max_y, (sty4 + 1) * HTS_ST4);
			if (b4_max_x <= b4_min_x || b4_max_y <= b4_min_y)
				continue;

			float u0, u1, v0, v1;
			htsBlockLocalRect(b4_min_x, b4_min_y, b4_max_x, b4_max_y, mean, u0, u1, v0, v1);

			if (st4_cand)
				atomicAdd((unsigned long long*)st4_cand, 1ULL);

			if (!htsRectMayContribute(a, b, c, q_thr, u0, u1, v0, v1))
				continue;

			if (st4_kept)
				atomicAdd((unsigned long long*)st4_kept, 1ULL);

			// 4x4-only mode: emit all tiles in surviving 4x4 block.
			if (hts_hier_mode == HTS_HIER_ST4_ONLY) {
				for (int ty = b4_min_y; ty < b4_max_y; ++ty)
					for (int tx = b4_min_x; tx < b4_max_x; ++tx)
						htsAcceptTile(do_emit, count, keys, values, inout_off, off_end,
							gauss_idx, depth_lo, grid_x, tx, ty, depth_sort_bits);
				continue;
			}

			// 4x4 -> 2x2 mid cull -> emit all tiles in surviving 2x2 block.
			const int st2_min_x = b4_min_x / HTS_ST2;
			const int st2_min_y = b4_min_y / HTS_ST2;
			const int st2_max_x = (b4_max_x + HTS_ST2 - 1) / HTS_ST2;
			const int st2_max_y = (b4_max_y + HTS_ST2 - 1) / HTS_ST2;

			for (int sty2 = st2_min_y; sty2 < st2_max_y; ++sty2) {
				for (int stx2 = st2_min_x; stx2 < st2_max_x; ++stx2) {
					const int b2_min_x = max(b4_min_x, stx2 * HTS_ST2);
					const int b2_max_x = min(b4_max_x, (stx2 + 1) * HTS_ST2);
					const int b2_min_y = max(b4_min_y, sty2 * HTS_ST2);
					const int b2_max_y = min(b4_max_y, (sty2 + 1) * HTS_ST2);
					if (b2_max_x <= b2_min_x || b2_max_y <= b2_min_y)
						continue;

					if (st2_cand)
						atomicAdd((unsigned long long*)st2_cand, 1ULL);

					htsBlockLocalRect(b2_min_x, b2_min_y, b2_max_x, b2_max_y, mean, u0, u1, v0, v1);
					if (!htsRectMayContribute(a, b, c, q_thr, u0, u1, v0, v1))
						continue;

					if (st2_kept)
						atomicAdd((unsigned long long*)st2_kept, 1ULL);

					for (int ty = b2_min_y; ty < b2_max_y; ++ty)
						for (int tx = b2_min_x; tx < b2_max_x; ++tx)
							htsAcceptTile(do_emit, count, keys, values, inout_off, off_end,
								gauss_idx, depth_lo, grid_x, tx, ty, depth_sort_bits);
				}
			}
		}
	}

	return count;
}

/** Count-only wrapper (no key writes). */
__device__ __forceinline__ uint32_t htsCountTiles(
	float2 mean,
	float a, float b, float c, float q_thr,
	int tile_min_x, int tile_min_y, int tile_max_x, int tile_max_y,
	int hts_mode,
	int hts_hier_mode)
{
	return htsProcessTiles(
		mean, a, b, c, q_thr,
		tile_min_x, tile_min_y, tile_max_x, tile_max_y,
		hts_mode,
		hts_hier_mode,
		24, // depth_sort_bits unused for count-only
		nullptr, nullptr, nullptr, nullptr,
		false,
		nullptr, nullptr, nullptr, 0,
		0, 0, 0);
}
