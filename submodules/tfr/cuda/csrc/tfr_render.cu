/*
 * 3DGS-style tile raster.
 * Eval: color only. Train: sample-bucket aux + PerGaussian backward (HST-GS-aligned).
 */
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cub/device/device_scan.cuh>
#include <cstdint>
#include <vector>

namespace cg = cooperative_groups;

namespace {

constexpr int BATCH = 256;  // BLOCK_SIZE
constexpr int SUB = 16;
constexpr int CHANNELS = 3;

__global__ void render_sorted_list_kernel(
    const uint2* __restrict__ ranges,
    const int32_t* __restrict__ point_list,
    const float* __restrict__ means2d,
    const float4* __restrict__ conic_opacity,
    const float* __restrict__ rgb,
    const float* __restrict__ bg_color,
    float* __restrict__ out_color,
    float* __restrict__ final_T,
    int32_t* __restrict__ n_contrib,
    int W,
    int H,
    int horizontal_blocks)
{
  auto block = cg::this_thread_block();
  uint32_t tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
  uint2 range = ranges[tile_id];

  uint2 pix_min = {block.group_index().x * SUB, block.group_index().y * SUB};
  uint2 pix_max = {min(pix_min.x + SUB, W), min(pix_min.y + SUB, H)};
  int pix_x = pix_min.x + (int)threadIdx.x;
  int pix_y = pix_min.y + (int)threadIdx.y;
  bool inside = pix_x < (int)pix_max.x && pix_y < (int)pix_max.y;
  bool done = !inside;
  float2 pixf = {(float)pix_x, (float)pix_y};
  int pix_id = pix_y * W + pix_x;

  int cnt = (int)range.y - (int)range.x;
  int toDo = cnt;
  int rounds = (cnt + BATCH - 1) / BATCH;

  __shared__ float2 collected_xy[BATCH];
  __shared__ float4 collected_conic_opacity[BATCH];
  __shared__ float3 collected_rgb[BATCH];

  float T = 1.0f;
  float C0 = 0.f, C1 = 0.f, C2 = 0.f;
  uint32_t contributor = 0;
  uint32_t last_contributor = 0;

  for (int r = 0; r < rounds; r++, toDo -= BATCH) {
    int num_done = __syncthreads_count(done);
    if (num_done == SUB * SUB)
      break;

    int progress = r * BATCH + block.thread_rank();
    int limit = min(BATCH, toDo);

    if (range.x + progress < range.y) {
      int gid = point_list[range.x + progress];
      collected_xy[block.thread_rank()] = make_float2(means2d[2 * gid], means2d[2 * gid + 1]);
      collected_conic_opacity[block.thread_rank()] = conic_opacity[gid];
      collected_rgb[block.thread_rank()] = make_float3(rgb[3 * gid], rgb[3 * gid + 1], rgb[3 * gid + 2]);
    }
    block.sync();

    for (int j = 0; !done && j < limit; ++j) {
      contributor++;
      float2 xy = collected_xy[j];
      float2 d = {xy.x - pixf.x, xy.y - pixf.y};
      float4 con_o = collected_conic_opacity[j];
      float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
      if (power > 0.0f)
        continue;
      float alpha = fminf(0.99f, con_o.w * expf(power));
      if (alpha < 1.0f / 255.0f)
        continue;
      float test_T = T * (1.0f - alpha);
      if (test_T < 0.0001f) {
        done = true;
        continue;
      }
      float3 c = collected_rgb[j];
      C0 += c.x * alpha * T;
      C1 += c.y * alpha * T;
      C2 += c.z * alpha * T;
      T = test_T;
      last_contributor = contributor;
    }
  }

  if (inside) {
    out_color[0 * H * W + pix_id] = C0 + T * bg_color[0];
    out_color[1 * H * W + pix_id] = C1 + T * bg_color[1];
    out_color[2 * H * W + pix_id] = C2 + T * bg_color[2];
    if (final_T)
      final_T[pix_id] = T;
    if (n_contrib)
      n_contrib[pix_id] = (int32_t)last_contributor;
  }
}

__global__ void per_tile_bucket_count_kernel(int T, const uint2* ranges, uint32_t* bucket_count)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T)
    return;
  uint2 range = ranges[idx];
  int num_splats = (int)range.y - (int)range.x;
  bucket_count[idx] = (uint32_t)((num_splats + 31) / 32);
}

// Train forward: write sample buckets every 32 Gaussians (golden-aligned).
__global__ void __launch_bounds__(BATCH) render_sorted_list_train_kernel(
    const uint2* __restrict__ ranges,
    const int32_t* __restrict__ point_list,
    const uint32_t* __restrict__ per_tile_bucket_offset,
    uint32_t* __restrict__ bucket_to_tile,
    __half* __restrict__ sampled_T,
    __half* __restrict__ sampled_ar,
    const float* __restrict__ means2d,
    const float4* __restrict__ conic_opacity,
    const float* __restrict__ rgb,
    const float* __restrict__ bg_color,
    float* __restrict__ out_color,
    float* __restrict__ final_T,
    int32_t* __restrict__ n_contrib,
    uint32_t* __restrict__ max_contrib,
    float* __restrict__ pixel_colors,
    int W,
    int H,
    int horizontal_blocks)
{
  auto block = cg::this_thread_block();
  uint32_t tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
  uint2 range = ranges[tile_id];

  uint2 pix_min = {block.group_index().x * SUB, block.group_index().y * SUB};
  uint2 pix_max = {min(pix_min.x + SUB, W), min(pix_min.y + SUB, H)};
  int pix_x = pix_min.x + (int)threadIdx.x;
  int pix_y = pix_min.y + (int)threadIdx.y;
  bool inside = pix_x < (int)pix_max.x && pix_y < (int)pix_max.y;
  bool done = !inside;
  float2 pixf = {(float)pix_x, (float)pix_y};
  int pix_id = pix_y * W + pix_x;

  int toDo = (int)range.y - (int)range.x;
  const int rounds = (toDo + BATCH - 1) / BATCH;

  uint32_t bbm = tile_id == 0 ? 0u : per_tile_bucket_offset[tile_id - 1];
  int num_buckets = (toDo + 31) / 32;
  for (int i = 0; i < (num_buckets + BATCH - 1) / BATCH; ++i) {
    int bucket_idx = i * BATCH + block.thread_rank();
    if (bucket_idx < num_buckets)
      bucket_to_tile[bbm + bucket_idx] = tile_id;
  }

  __shared__ int collected_id[BATCH];
  __shared__ float2 collected_xy[BATCH];
  __shared__ float4 collected_conic_opacity[BATCH];

  float T = 1.0f;
  uint32_t contributor = 0;
  uint32_t last_contributor = 0;
  float C[CHANNELS] = {0, 0, 0};

  for (int i = 0; i < rounds; i++, toDo -= BATCH) {
    int num_done = __syncthreads_count(done);
    if (num_done == BATCH)
      break;

    int progress = i * BATCH + block.thread_rank();
    if (range.x + progress < range.y) {
      int coll_id = point_list[range.x + progress];
      collected_id[block.thread_rank()] = coll_id;
      collected_xy[block.thread_rank()] = make_float2(means2d[2 * coll_id], means2d[2 * coll_id + 1]);
      collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
    }
    block.sync();

    for (int j = 0; !done && j < min(BATCH, toDo); ++j) {
      if (j % 32 == 0) {
        sampled_T[(bbm * BATCH) + block.thread_rank()] = __float2half_rn(T);
        for (int ch = 0; ch < CHANNELS; ++ch)
          sampled_ar[(bbm * BATCH * CHANNELS) + ch * BATCH + block.thread_rank()] = __float2half_rn(C[ch]);
        ++bbm;
      }

      contributor++;
      float2 xy = collected_xy[j];
      float2 d = {xy.x - pixf.x, xy.y - pixf.y};
      float4 con_o = collected_conic_opacity[j];
      float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
      if (power > 0.0f)
        continue;
      float alpha = fminf(0.99f, con_o.w * expf(power));
      if (alpha < 1.0f / 255.0f)
        continue;
      float test_T = T * (1.0f - alpha);
      if (test_T < 0.0001f) {
        done = true;
        continue;
      }
      int gid = collected_id[j];
      for (int ch = 0; ch < CHANNELS; ++ch)
        C[ch] += rgb[gid * CHANNELS + ch] * alpha * T;
      T = test_T;
      last_contributor = contributor;
    }
  }

  if (inside) {
    final_T[pix_id] = T;
    n_contrib[pix_id] = (int32_t)last_contributor;
    for (int ch = 0; ch < CHANNELS; ++ch) {
      pixel_colors[ch * H * W + pix_id] = C[ch];
      out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
    }
  }

  __shared__ uint32_t block_max_contributor;
  if (block.thread_rank() == 0)
    block_max_contributor = 0;
  block.sync();
  atomicMax(&block_max_contributor, last_contributor);
  block.sync();
  if (block.thread_rank() == 0)
    max_contrib[tile_id] = block_max_contributor;
}

// PerGaussian backward (ported from golden/HST-GS).
__global__ void PerGaussianRenderCUDA(
    const uint2* __restrict__ ranges,
    const int32_t* __restrict__ point_list,
    int W,
    int H,
    int B,
    const uint32_t* __restrict__ per_tile_bucket_offset,
    const uint32_t* __restrict__ bucket_to_tile,
    const __half* __restrict__ sampled_T,
    const __half* __restrict__ sampled_ar,
    const float* __restrict__ bg_color,
    const float* __restrict__ means2d,
    const float4* __restrict__ conic_opacity,
    const float* __restrict__ colors,
    const float* __restrict__ final_Ts,
    const int32_t* __restrict__ n_contrib,
    const uint32_t* __restrict__ max_contrib,
    const float* __restrict__ pixel_colors,
    const float* __restrict__ dL_dpixels,
    float4* __restrict__ dL_dmean2D,
    float4* __restrict__ dL_dconic2D,
    float* __restrict__ dL_dopacity,
    float* __restrict__ dL_dcolors)
{
  auto block = cg::this_thread_block();
  auto my_warp = cg::tiled_partition<32>(block);
  const unsigned int lane = my_warp.thread_rank();
  uint32_t global_bucket_idx = block.group_index().x * my_warp.meta_group_size() + my_warp.meta_group_rank();
  if (global_bucket_idx >= (uint32_t)B)
    return;

  uint32_t tile_id = bucket_to_tile[global_bucket_idx];
  uint2 range = ranges[tile_id];
  int num_splats_in_tile = (int)range.y - (int)range.x;
  uint32_t bbm = tile_id == 0 ? 0u : per_tile_bucket_offset[tile_id - 1];
  int bucket_idx_in_tile = (int)global_bucket_idx - (int)bbm;
  int splat_idx_in_tile = bucket_idx_in_tile * 32 + (int)my_warp.thread_rank();
  int splat_idx_global = (int)range.x + splat_idx_in_tile;
  bool valid_splat = splat_idx_in_tile < num_splats_in_tile;
  if (bucket_idx_in_tile * 32 >= (int)max_contrib[tile_id])
    return;

  int gaussian_idx = 0;
  float2 xy = {0.0f, 0.0f};
  float4 con_o = {0.0f, 0.0f, 0.0f, 0.0f};
  float c[CHANNELS] = {0.0f};
  if (valid_splat) {
    gaussian_idx = point_list[splat_idx_global];
    xy = make_float2(means2d[2 * gaussian_idx], means2d[2 * gaussian_idx + 1]);
    con_o = conic_opacity[gaussian_idx];
    for (int ch = 0; ch < CHANNELS; ++ch)
      c[ch] = colors[gaussian_idx * CHANNELS + ch];
  }

  float reg_mean_x = 0.0f, reg_mean_y = 0.0f;
  float reg_mean_abs_x = 0.0f, reg_mean_abs_y = 0.0f;
  float reg_conic_x = 0.0f, reg_conic_y = 0.0f, reg_conic_w = 0.0f;
  float reg_opacity = 0.0f;
  float reg_colors[CHANNELS] = {0.0f};

  const uint32_t horizontal_blocks = (W + SUB - 1) / SUB;
  const uint2 tile = {tile_id % horizontal_blocks, tile_id / horizontal_blocks};
  const uint2 pix_min = {tile.x * SUB, tile.y * SUB};

  float T = 0.0f, T_final = 0.0f, last_contributor = 0.0f;
  float ar[CHANNELS] = {0.0f};
  float dL_dpixel[CHANNELS] = {0.0f};
  const float ddelx_dx = 0.5f * W;
  const float ddely_dy = 0.5f * H;

  __shared__ float Shared_sampled_ar[32 * CHANNELS + 1];
  const __half* sampled_ar_b = sampled_ar + global_bucket_idx * BATCH * CHANNELS;
  __shared__ float Shared_pixels[32 * CHANNELS];

#pragma unroll
  for (int i = 0; i < BATCH + 31; ++i) {
    if (i % 32 == 0) {
      for (int ch = 0; ch < CHANNELS; ++ch) {
        int shift = BATCH * ch + i + block.thread_rank();
        Shared_sampled_ar[ch * 32 + lane] = 0.0f;
        if (i + (int)block.thread_rank() < BATCH)
          Shared_sampled_ar[ch * 32 + lane] = __half2float(sampled_ar_b[shift]);
      }
      const uint32_t local_id = i + block.thread_rank();
      const uint2 pix = {pix_min.x + local_id % SUB, pix_min.y + local_id / SUB};
      for (int ch = 0; ch < CHANNELS; ++ch)
        Shared_pixels[ch * 32 + lane] = 0.0f;
      if (local_id < BATCH && pix.x < (uint32_t)W && pix.y < (uint32_t)H) {
        const uint32_t id = W * pix.y + pix.x;
        for (int ch = 0; ch < CHANNELS; ++ch)
          Shared_pixels[ch * 32 + lane] = pixel_colors[ch * H * W + id];
      }
      block.sync();
    }

    T = my_warp.shfl_up(T, 1);
    last_contributor = my_warp.shfl_up(last_contributor, 1);
    T_final = my_warp.shfl_up(T_final, 1);
    for (int ch = 0; ch < CHANNELS; ++ch) {
      ar[ch] = my_warp.shfl_up(ar[ch], 1);
      dL_dpixel[ch] = my_warp.shfl_up(dL_dpixel[ch], 1);
    }

    int idx = i - (int)my_warp.thread_rank();
    const uint2 pix = {pix_min.x + idx % SUB, pix_min.y + idx / SUB};
    const uint32_t pix_id = W * pix.y + pix.x;
    const float2 pixf = {(float)pix.x, (float)pix.y};
    bool valid_pixel = pix.x < (uint32_t)W && pix.y < (uint32_t)H;

    if (valid_splat && valid_pixel && my_warp.thread_rank() == 0 && idx < BATCH) {
      T = __half2float(sampled_T[global_bucket_idx * BATCH + idx]);
      int ii = i % 32;
      for (int ch = 0; ch < CHANNELS; ++ch)
        ar[ch] = -Shared_pixels[ch * 32 + ii] + Shared_sampled_ar[ch * 32 + ii];
      T_final = final_Ts[pix_id];
      last_contributor = (float)n_contrib[pix_id];
      for (int ch = 0; ch < CHANNELS; ++ch)
        dL_dpixel[ch] = dL_dpixels[ch * H * W + pix_id];
    }

    if (valid_splat && valid_pixel && 0 <= idx && idx < BATCH) {
      if (splat_idx_in_tile >= (int)last_contributor)
        continue;
      const float2 d = {xy.x - pixf.x, xy.y - pixf.y};
      const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
      if (power > 0.0f)
        continue;
      const float G = __expf(power);
      const float alpha = fminf(0.99f, con_o.w * G);
      if (alpha < 1.0f / 255.0f)
        continue;
      const float dchannel_dcolor = alpha * T;
      const float one_minus_alpha_reci = 1.0f / (1.0f - alpha);

      float dL_dalpha = 0.0f;
      float bg_dot_dpixel = 0.0f;
      for (int ch = 0; ch < CHANNELS; ++ch) {
        ar[ch] += dchannel_dcolor * c[ch];
        const float& dL_dchannel = dL_dpixel[ch];
        reg_colors[ch] += dchannel_dcolor * dL_dchannel;
        dL_dalpha += (c[ch] * T + one_minus_alpha_reci * ar[ch]) * dL_dchannel;
        bg_dot_dpixel += bg_color[ch] * dL_dchannel;
      }
      dL_dalpha += (-T_final * one_minus_alpha_reci) * bg_dot_dpixel;
      T *= (1.0f - alpha);

      const float dL_dG = con_o.w * dL_dalpha;
      const float gdx = G * d.x;
      const float gdy = G * d.y;
      const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
      const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;
      const float tmp_x = dL_dG * dG_ddelx * ddelx_dx;
      const float tmp_y = dL_dG * dG_ddely * ddely_dy;
      reg_mean_x += tmp_x;
      reg_mean_y += tmp_y;
      reg_mean_abs_x += fabsf(tmp_x);
      reg_mean_abs_y += fabsf(tmp_y);
      reg_conic_x += -0.5f * gdx * d.x * dL_dG;
      reg_conic_y += -0.5f * gdx * d.y * dL_dG;
      reg_conic_w += -0.5f * gdy * d.y * dL_dG;
      reg_opacity += G * dL_dalpha;
    }
  }

  if (valid_splat) {
    const unsigned int active_mask = __activemask();
    const unsigned int group_mask = __match_any_sync(active_mask, (unsigned)gaussian_idx);
    const int leader = __ffs(group_mask) - 1;
    const int group_size = __popc(group_mask);

    if (group_size == 1) {
      if ((int)lane == leader) {
        atomicAdd(&dL_dmean2D[gaussian_idx].x, reg_mean_x);
        atomicAdd(&dL_dmean2D[gaussian_idx].y, reg_mean_y);
        atomicAdd(&dL_dmean2D[gaussian_idx].z, reg_mean_abs_x);
        atomicAdd(&dL_dmean2D[gaussian_idx].w, reg_mean_abs_y);
        atomicAdd(&dL_dconic2D[gaussian_idx].x, reg_conic_x);
        atomicAdd(&dL_dconic2D[gaussian_idx].y, reg_conic_y);
        atomicAdd(&dL_dconic2D[gaussian_idx].w, reg_conic_w);
        atomicAdd(&dL_dopacity[gaussian_idx], reg_opacity);
        for (int ch = 0; ch < CHANNELS; ++ch)
          atomicAdd(&dL_dcolors[gaussian_idx * CHANNELS + ch], reg_colors[ch]);
      }
    } else {
      auto warp_group_sum = [&](float v) {
        float sum = v;
        for (int offset = 16; offset > 0; offset >>= 1) {
          float other = __shfl_down_sync(group_mask, sum, offset);
          unsigned int src_lane = lane + offset;
          if (src_lane < 32 && (group_mask & (1u << src_lane)))
            sum += other;
        }
        return sum;
      };

      const float mean_x_sum = warp_group_sum(reg_mean_x);
      const float mean_y_sum = warp_group_sum(reg_mean_y);
      const float mean_abs_x_sum = warp_group_sum(reg_mean_abs_x);
      const float mean_abs_y_sum = warp_group_sum(reg_mean_abs_y);
      const float conic_x_sum = warp_group_sum(reg_conic_x);
      const float conic_y_sum = warp_group_sum(reg_conic_y);
      const float conic_w_sum = warp_group_sum(reg_conic_w);
      const float opacity_sum = warp_group_sum(reg_opacity);

      if ((int)lane == leader) {
        atomicAdd(&dL_dmean2D[gaussian_idx].x, mean_x_sum);
        atomicAdd(&dL_dmean2D[gaussian_idx].y, mean_y_sum);
        atomicAdd(&dL_dmean2D[gaussian_idx].z, mean_abs_x_sum);
        atomicAdd(&dL_dmean2D[gaussian_idx].w, mean_abs_y_sum);
        atomicAdd(&dL_dconic2D[gaussian_idx].x, conic_x_sum);
        atomicAdd(&dL_dconic2D[gaussian_idx].y, conic_y_sum);
        atomicAdd(&dL_dconic2D[gaussian_idx].w, conic_w_sum);
        atomicAdd(&dL_dopacity[gaussian_idx], opacity_sum);
      }
      for (int ch = 0; ch < CHANNELS; ++ch) {
        const float color_sum = warp_group_sum(reg_colors[ch]);
        if ((int)lane == leader)
          atomicAdd(&dL_dcolors[gaussian_idx * CHANNELS + ch], color_sum);
      }
    }
  }
}

} // namespace

torch::Tensor render_sorted_list(
    torch::Tensor ranges,
    torch::Tensor point_list,
    torch::Tensor means2d,
    torch::Tensor conic_opacity,
    torch::Tensor rgb,
    torch::Tensor bg_color,
    int64_t W,
    int64_t H,
    int64_t tiles_x,
    int64_t tiles_y)
{
  TORCH_CHECK(ranges.is_cuda(), "tensors must be CUDA");
  int horizontal_blocks = (int)tiles_x;
  auto bg = bg_color.reshape({3, 1, 1});
  auto out = bg.expand({3, (int)H, (int)W}).contiguous().clone();

  dim3 grid(horizontal_blocks, (int)tiles_y, 1);
  dim3 block(SUB, SUB, 1);
  render_sorted_list_kernel<<<grid, block>>>(
      reinterpret_cast<const uint2*>(ranges.data_ptr<int32_t>()),
      point_list.data_ptr<int32_t>(),
      means2d.data_ptr<float>(),
      reinterpret_cast<const float4*>(conic_opacity.data_ptr<float>()),
      rgb.data_ptr<float>(),
      bg_color.data_ptr<float>(),
      out.data_ptr<float>(),
      nullptr,
      nullptr,
      (int)W,
      (int)H,
      horizontal_blocks);
  return out;
}

std::vector<torch::Tensor> render_sorted_list_train(
    torch::Tensor ranges,
    torch::Tensor point_list,
    torch::Tensor means2d,
    torch::Tensor conic_opacity,
    torch::Tensor rgb,
    torch::Tensor bg_color,
    int64_t W,
    int64_t H,
    int64_t tiles_x,
    int64_t tiles_y)
{
  TORCH_CHECK(ranges.is_cuda(), "tensors must be CUDA");
  int num_tiles = (int)(tiles_x * tiles_y);
  int horizontal_blocks = (int)tiles_x;
  auto opts = means2d.options();
  auto opts_i32 = means2d.options().dtype(torch::kInt32);
  auto opts_u32 = means2d.options().dtype(torch::kInt32);

  auto bg = bg_color.reshape({3, 1, 1});
  auto out = bg.expand({3, (int)H, (int)W}).contiguous().clone();
  auto final_T = torch::zeros({(int)H * (int)W}, opts);
  auto n_contrib = torch::zeros({(int)H * (int)W}, opts_i32);
  auto max_contrib = torch::zeros({num_tiles}, opts_u32);
  auto pixel_colors = torch::zeros({3, (int)H, (int)W}, opts);

  auto bucket_count = torch::empty({num_tiles}, opts_u32);
  {
    int threads = 256;
    int blocks = (num_tiles + threads - 1) / threads;
    per_tile_bucket_count_kernel<<<blocks, threads>>>(
        num_tiles,
        reinterpret_cast<const uint2*>(ranges.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(bucket_count.data_ptr<int32_t>()));
  }

  auto per_tile_bucket_offset = torch::empty({num_tiles}, opts_u32);
  size_t scan_bytes = 0;
  cub::DeviceScan::InclusiveSum(
      nullptr, scan_bytes,
      reinterpret_cast<uint32_t*>(bucket_count.data_ptr<int32_t>()),
      reinterpret_cast<uint32_t*>(per_tile_bucket_offset.data_ptr<int32_t>()),
      num_tiles);
  auto scan_ws = torch::empty({(int64_t)scan_bytes}, means2d.options().dtype(torch::kUInt8));
  cub::DeviceScan::InclusiveSum(
      scan_ws.data_ptr<uint8_t>(), scan_bytes,
      reinterpret_cast<uint32_t*>(bucket_count.data_ptr<int32_t>()),
      reinterpret_cast<uint32_t*>(per_tile_bucket_offset.data_ptr<int32_t>()),
      num_tiles);

  int num_buckets = 0;
  if (num_tiles > 0) {
    cudaMemcpy(&num_buckets, per_tile_bucket_offset.data_ptr<int32_t>() + num_tiles - 1,
               sizeof(int), cudaMemcpyDeviceToHost);
  }

  auto bucket_to_tile = torch::zeros({std::max(num_buckets, 1)}, opts_u32);
  auto sampled_T = torch::zeros({std::max(num_buckets, 1) * BATCH}, means2d.options().dtype(torch::kFloat16));
  auto sampled_ar =
      torch::zeros({std::max(num_buckets, 1) * BATCH * CHANNELS}, means2d.options().dtype(torch::kFloat16));

  dim3 grid(horizontal_blocks, (int)tiles_y, 1);
  dim3 block(SUB, SUB, 1);
  render_sorted_list_train_kernel<<<grid, block>>>(
      reinterpret_cast<const uint2*>(ranges.data_ptr<int32_t>()),
      point_list.data_ptr<int32_t>(),
      reinterpret_cast<const uint32_t*>(per_tile_bucket_offset.data_ptr<int32_t>()),
      reinterpret_cast<uint32_t*>(bucket_to_tile.data_ptr<int32_t>()),
      reinterpret_cast<__half*>(sampled_T.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(sampled_ar.data_ptr<at::Half>()),
      means2d.data_ptr<float>(),
      reinterpret_cast<const float4*>(conic_opacity.data_ptr<float>()),
      rgb.data_ptr<float>(),
      bg_color.data_ptr<float>(),
      out.data_ptr<float>(),
      final_T.data_ptr<float>(),
      n_contrib.data_ptr<int32_t>(),
      reinterpret_cast<uint32_t*>(max_contrib.data_ptr<int32_t>()),
      pixel_colors.data_ptr<float>(),
      (int)W,
      (int)H,
      horizontal_blocks);

  auto num_buckets_t = torch::tensor({num_buckets}, means2d.options().dtype(torch::kInt64));
  return {out, final_T, n_contrib, max_contrib, pixel_colors, per_tile_bucket_offset,
          bucket_to_tile, sampled_T, sampled_ar, num_buckets_t};
}

std::vector<torch::Tensor> render_sorted_list_backward(
    torch::Tensor ranges,
    torch::Tensor point_list,
    torch::Tensor means2d,
    torch::Tensor conic_opacity,
    torch::Tensor rgb,
    torch::Tensor bg_color,
    torch::Tensor final_T,
    torch::Tensor n_contrib,
    torch::Tensor max_contrib,
    torch::Tensor pixel_colors,
    torch::Tensor per_tile_bucket_offset,
    torch::Tensor bucket_to_tile,
    torch::Tensor sampled_T,
    torch::Tensor sampled_ar,
    torch::Tensor dL_dpixels,
    int64_t W,
    int64_t H,
    int64_t tiles_x,
    int64_t tiles_y,
    int64_t num_buckets)
{
  int P = (int)means2d.size(0);
  auto opts = means2d.options();
  auto dL_dmean2D = torch::zeros({P, 4}, opts);
  auto dL_dconic2D = torch::zeros({P, 4}, opts);
  auto dL_dopacity = torch::zeros({P, 1}, opts);
  auto dL_dcolors = torch::zeros({P, 3}, opts);

  int B = (int)num_buckets;
  if (B > 0) {
    const int THREADS = 32;
    PerGaussianRenderCUDA<<<((B * 32) + THREADS - 1) / THREADS, THREADS>>>(
        reinterpret_cast<const uint2*>(ranges.data_ptr<int32_t>()),
        point_list.data_ptr<int32_t>(),
        (int)W,
        (int)H,
        B,
        reinterpret_cast<const uint32_t*>(per_tile_bucket_offset.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(bucket_to_tile.data_ptr<int32_t>()),
        reinterpret_cast<const __half*>(sampled_T.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(sampled_ar.data_ptr<at::Half>()),
        bg_color.data_ptr<float>(),
        means2d.data_ptr<float>(),
        reinterpret_cast<const float4*>(conic_opacity.data_ptr<float>()),
        rgb.data_ptr<float>(),
        final_T.data_ptr<float>(),
        n_contrib.data_ptr<int32_t>(),
        reinterpret_cast<const uint32_t*>(max_contrib.data_ptr<int32_t>()),
        pixel_colors.data_ptr<float>(),
        dL_dpixels.data_ptr<float>(),
        reinterpret_cast<float4*>(dL_dmean2D.data_ptr<float>()),
        reinterpret_cast<float4*>(dL_dconic2D.data_ptr<float>()),
        dL_dopacity.data_ptr<float>(),
        dL_dcolors.data_ptr<float>());
  }
  return {dL_dmean2D, dL_dconic2D, dL_dopacity, dL_dcolors};
}
