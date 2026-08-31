/*
 * 3DGS-style binning: tiles_touched → prefix sum → duplicate keys → CUB radix sort → tile ranges.
 */
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cstdint>
#include <vector>

namespace {

constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;
constexpr int DEPTH_SORT_BITS = 24;
constexpr int DEPTH_SORT_SHIFT = 32 - DEPTH_SORT_BITS;

static uint32_t getHigherMsbHost(uint32_t n)
{
  uint32_t msb = 0;
  while (n > 0) {
    n >>= 1;
    msb++;
  }
  return msb;
}

// tiles_touched derived from preprocess-cached tile_bounds (no extent recompute).
__global__ void tiles_touched_from_bounds_kernel(
    const int4* __restrict__ tile_bounds,
    const int32_t* __restrict__ radii,
    int32_t* __restrict__ tiles_touched,
    int P)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= P) return;
  tiles_touched[idx] = 0;
  if (radii[idx] <= 0) return;
  const int4 b = tile_bounds[idx];
  const int w = b.z - b.x;
  const int h = b.w - b.y;
  if (w <= 0 || h <= 0) return;
  tiles_touched[idx] = w * h;
}

// Golden-style duplicateWithKeys; reads preprocess-cached tile_bounds.
__global__ void duplicate_with_keys_kernel(
    int P,
    const float* __restrict__ depths,
    const int4* __restrict__ tile_bounds,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ radii,
    uint64_t* __restrict__ keys_unsorted,
    uint32_t* __restrict__ values_unsorted,
    int grid_x,
    int grid_y)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= P) return;
  if (radii[idx] <= 0) return;

  uint32_t off = (idx == 0) ? 0u : (uint32_t)offsets[idx - 1];
  const uint32_t off_end = (uint32_t)offsets[idx];
  const uint64_t invalid_key = ((uint64_t)0xFFFFFFFFu) << DEPTH_SORT_BITS;

  const int4 b = tile_bounds[idx];
  const int tile_min_x = b.x;
  const int tile_min_y = b.y;
  const int tile_max_x = b.z;
  const int tile_max_y = b.w;
  if (tile_max_x <= tile_min_x || tile_max_y <= tile_min_y) return;

  for (int y = tile_min_y; y < tile_max_y; ++y) {
    for (int x = tile_min_x; x < tile_max_x; ++x) {
      if (off >= off_end) break;
      uint64_t key = (uint64_t)(y * grid_x + x);
      key <<= DEPTH_SORT_BITS;
      key |= (uint64_t)(__float_as_uint(depths[idx]) >> DEPTH_SORT_SHIFT);
      keys_unsorted[off] = key;
      values_unsorted[off] = (uint32_t)idx;
      off++;
    }
    if (off >= off_end) break;
  }
  while (off < off_end) {
    keys_unsorted[off] = invalid_key;
    values_unsorted[off] = 0;
    off++;
  }
}

__global__ void identify_tile_ranges_kernel(
    int L,
    const uint64_t* __restrict__ point_list_keys,
    uint2* __restrict__ ranges)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= L) return;

  uint64_t key = point_list_keys[idx];
  uint32_t currtile = (uint32_t)(key >> DEPTH_SORT_BITS);
  const bool curr_valid = currtile != 0xFFFFFFFFu;
  if (idx == 0) {
    if (curr_valid) ranges[currtile].x = 0;
  } else {
    uint32_t prevtile = (uint32_t)(point_list_keys[idx - 1] >> DEPTH_SORT_BITS);
    const bool prev_valid = prevtile != 0xFFFFFFFFu;
    if (currtile != prevtile) {
      if (prev_valid) ranges[prevtile].y = (uint32_t)idx;
      if (curr_valid) ranges[currtile].x = (uint32_t)idx;
    }
  }
  if (idx == L - 1 && curr_valid)
    ranges[currtile].y = (uint32_t)L;
}

// Count tiles whose range is non-empty (end > begin). ranges is pre-zeroed, so
// empty tiles stay (0,0); any tile with at least one gaussian has end > begin.
__global__ void count_nonempty_tiles_kernel(
    const uint2* __restrict__ ranges,
    int num_tiles,
    int32_t* __restrict__ out_count)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_tiles) return;
  if (ranges[idx].y > ranges[idx].x)
    atomicAdd(out_count, 1);
}

} // namespace

namespace {

// Grow-only scratch buffers so we skip CUB size queries on steady-state frames.
struct BinningScratch {
  int cap_P = 0;
  int cap_L = 0;
  int cap_tiles = 0;
  size_t scan_bytes = 0;
  size_t sort_bytes = 0;
  torch::Tensor tiles_touched;
  torch::Tensor point_offsets;
  torch::Tensor scan_workspace;
  torch::Tensor sort_workspace;
  torch::Tensor keys_unsorted;
  torch::Tensor keys_sorted;
  torch::Tensor vals_unsorted;
  torch::Tensor vals_sorted;
  torch::Tensor ranges;
};

BinningScratch& scratch() {
  static BinningScratch s;
  return s;
}

void ensure_P(BinningScratch& s, int P, const torch::TensorOptions& opts_i32, const torch::TensorOptions& opts_u8) {
  if (P <= s.cap_P) return;
  s.cap_P = P;
  s.tiles_touched = torch::empty({P}, opts_i32);
  s.point_offsets = torch::empty({P}, opts_i32);
  size_t need = 0;
  cub::DeviceScan::InclusiveSum(
      nullptr, need,
      s.tiles_touched.data_ptr<int32_t>(),
      s.point_offsets.data_ptr<int32_t>(),
      P);
  s.scan_bytes = need;
  s.scan_workspace = torch::empty({(int64_t)need}, opts_u8);
}

void ensure_L(BinningScratch& s, int L, const torch::TensorOptions& opts_i64, const torch::TensorOptions& opts_i32, const torch::TensorOptions& opts_u8) {
  if (L <= s.cap_L) return;
  // Over-allocate ~12.5% to reduce realloc churn across views.
  int cap = L + L / 8 + 1024;
  s.cap_L = cap;
  s.keys_unsorted = torch::empty({cap}, opts_i64);
  s.keys_sorted = torch::empty({cap}, opts_i64);
  s.vals_unsorted = torch::empty({cap}, opts_i32);
  s.vals_sorted = torch::empty({cap}, opts_i32);
  size_t need = 0;
  cub::DeviceRadixSort::SortPairs(
      nullptr, need,
      reinterpret_cast<uint64_t*>(s.keys_unsorted.data_ptr<int64_t>()),
      reinterpret_cast<uint64_t*>(s.keys_sorted.data_ptr<int64_t>()),
      reinterpret_cast<uint32_t*>(s.vals_unsorted.data_ptr<int32_t>()),
      reinterpret_cast<uint32_t*>(s.vals_sorted.data_ptr<int32_t>()),
      cap);
  s.sort_bytes = need;
  s.sort_workspace = torch::empty({(int64_t)need}, opts_u8);
}

void ensure_tiles(BinningScratch& s, int num_tiles, const torch::TensorOptions& opts_i32) {
  if (num_tiles <= s.cap_tiles) return;
  s.cap_tiles = num_tiles;
  s.ranges = torch::empty({num_tiles, 2}, opts_i32);
}

} // namespace

torch::Tensor render_sorted_list(
    torch::Tensor ranges,       // (num_tiles, 2) int32 begin/end
    torch::Tensor point_list,   // (L,) int32 sorted gaussian ids
    torch::Tensor means2d,
    torch::Tensor conic_opacity,
    torch::Tensor rgb,
    torch::Tensor bg_color,
    int64_t W,
    int64_t H,
    int64_t tiles_x,
    int64_t tiles_y);

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
    int64_t tiles_y);

std::vector<torch::Tensor> preprocess_gaussians(
    torch::Tensor means3D,
    torch::Tensor scales,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor sh,
    torch::Tensor viewmatrix,
    torch::Tensor projmatrix,
    torch::Tensor campos,
    int64_t W,
    int64_t H,
    double tan_fovx,
    double tan_fovy,
    int64_t sh_degree,
    double scale_modifier,
    int64_t activate_params,
    torch::Tensor sh_rest);

std::vector<torch::Tensor> forward_packed(
    torch::Tensor means3D,
    torch::Tensor scales,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor sh,
    torch::Tensor viewmatrix,
    torch::Tensor projmatrix,
    torch::Tensor campos,
    torch::Tensor bg_color,
    int64_t W,
    int64_t H,
    double tan_fovx,
    double tan_fovy,
    int64_t sh_degree,
    double scale_modifier,
    int64_t activate_params,
    torch::Tensor sh_rest)
{
  TORCH_CHECK(means3D.is_cuda(), "inputs must be CUDA");
  int P = (int)means3D.size(0);
  int grid_x = (int)((W + BLOCK_X - 1) / BLOCK_X);
  int grid_y = (int)((H + BLOCK_Y - 1) / BLOCK_Y);
  int num_tiles = grid_x * grid_y;

  auto pre = preprocess_gaussians(
      means3D, scales, rotations, opacities, sh,
      viewmatrix, projmatrix, campos,
      W, H, tan_fovx, tan_fovy, sh_degree, scale_modifier, activate_params, sh_rest);

  auto means2d = pre[0];
  auto depths = pre[1];
  auto conic_opacity = pre[2];
  auto rgb = pre[3];
  auto radii = pre[6];
  auto tile_bounds = pre[10];

  auto device = means3D.device();
  auto opts_i32 = means3D.options().dtype(torch::kInt32);
  auto opts_i64 = means3D.options().dtype(torch::kInt64);
  auto opts_u8 = means3D.options().dtype(torch::kUInt8);

  auto bg = bg_color.to(device).to(torch::kFloat32).contiguous();
  if (bg.numel() == 1) bg = bg.expand({3}).contiguous();

  auto& s = scratch();
  ensure_P(s, P, opts_i32, opts_u8);
  // tiles_touched must be zeroed each frame
  s.tiles_touched.narrow(0, 0, P).zero_();
  {
    int threads = 256;
    int blocks = (P + threads - 1) / threads;
    tiles_touched_from_bounds_kernel<<<blocks, threads>>>(
        reinterpret_cast<const int4*>(tile_bounds.data_ptr<int32_t>()),
        radii.data_ptr<int32_t>(),
        s.tiles_touched.data_ptr<int32_t>(),
        P);
  }

  cub::DeviceScan::InclusiveSum(
      s.scan_workspace.data_ptr<uint8_t>(), s.scan_bytes,
      s.tiles_touched.data_ptr<int32_t>(),
      s.point_offsets.data_ptr<int32_t>(),
      P);

  int num_rendered = 0;
  if (P > 0) {
    cudaMemcpy(&num_rendered, s.point_offsets.data_ptr<int32_t>() + P - 1,
               sizeof(int), cudaMemcpyDeviceToHost);
  }

  // Host-side scalars avoid extra CUDA tensor alloc / sync for stats.
  auto T_t = torch::tensor(int64_t(num_tiles), torch::dtype(torch::kInt64));
  auto L_t = torch::tensor(int64_t(num_rendered), torch::dtype(torch::kInt64));

  if (num_rendered <= 0) {
    // No gaussian touches any tile -> zero non-empty tiles.
    auto zero_tiles = torch::tensor(int64_t(0), torch::dtype(torch::kInt64));
    auto out = bg.view({3, 1, 1}).expand({3, (int)H, (int)W}).contiguous().clone();
    return {out, radii, zero_tiles, L_t};
  }

  ensure_L(s, num_rendered, opts_i64, opts_i32, opts_u8);
  ensure_tiles(s, num_tiles, opts_i32);

  {
    int threads = 256;
    int blocks = (P + threads - 1) / threads;
    duplicate_with_keys_kernel<<<blocks, threads>>>(
        P,
        depths.data_ptr<float>(),
        reinterpret_cast<const int4*>(tile_bounds.data_ptr<int32_t>()),
        s.point_offsets.data_ptr<int32_t>(),
        radii.data_ptr<int32_t>(),
        reinterpret_cast<uint64_t*>(s.keys_unsorted.data_ptr<int64_t>()),
        reinterpret_cast<uint32_t*>(s.vals_unsorted.data_ptr<int32_t>()),
        grid_x, grid_y);
  }

  int bit = (int)getHigherMsbHost((uint32_t)num_tiles);
  cub::DeviceRadixSort::SortPairs(
      s.sort_workspace.data_ptr<uint8_t>(), s.sort_bytes,
      reinterpret_cast<uint64_t*>(s.keys_unsorted.data_ptr<int64_t>()),
      reinterpret_cast<uint64_t*>(s.keys_sorted.data_ptr<int64_t>()),
      reinterpret_cast<uint32_t*>(s.vals_unsorted.data_ptr<int32_t>()),
      reinterpret_cast<uint32_t*>(s.vals_sorted.data_ptr<int32_t>()),
      num_rendered, 0, DEPTH_SORT_BITS + bit);

  s.ranges.narrow(0, 0, num_tiles).zero_();
  {
    int threads = 256;
    int blocks = (num_rendered + threads - 1) / threads;
    identify_tile_ranges_kernel<<<blocks, threads>>>(
        num_rendered,
        reinterpret_cast<const uint64_t*>(s.keys_sorted.data_ptr<int64_t>()),
        reinterpret_cast<uint2*>(s.ranges.data_ptr<int32_t>()));
  }

  // Pass views of the used prefix so render does not walk past L.
  auto ranges_view = s.ranges.narrow(0, 0, num_tiles);
  auto vals_view = s.vals_sorted.narrow(0, 0, num_rendered);
  auto image = render_sorted_list(
      ranges_view, vals_view, means2d, conic_opacity, rgb, bg, W, H, grid_x, grid_y);

  // active_tiles = number of non-empty tiles (end > begin), not the screen tile
  // total. One tiny device counter per frame.
  auto active_tiles = T_t;  // fallback (screen total) if counting is skipped
  if (num_tiles > 0) {
    auto counter = torch::zeros({1}, opts_i32);
    int threads = 256;
    int blocks = (num_tiles + threads - 1) / threads;
    count_nonempty_tiles_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint2*>(s.ranges.data_ptr<int32_t>()),
        num_tiles,
        counter.data_ptr<int32_t>());
    active_tiles = counter.to(torch::kInt64).cpu();
  }

  return {image, radii, active_tiles, L_t};
}

// Differentiable forward: clone binning buffers + final_T/n_contrib for backward.
std::vector<torch::Tensor> forward_packed_train(
    torch::Tensor means3D,
    torch::Tensor scales,
    torch::Tensor rotations,
    torch::Tensor opacities,
    torch::Tensor sh,
    torch::Tensor viewmatrix,
    torch::Tensor projmatrix,
    torch::Tensor campos,
    torch::Tensor bg_color,
    int64_t W,
    int64_t H,
    double tan_fovx,
    double tan_fovy,
    int64_t sh_degree,
    double scale_modifier,
    int64_t activate_params,
    torch::Tensor sh_rest)
{
  TORCH_CHECK(means3D.is_cuda(), "inputs must be CUDA");
  int P = (int)means3D.size(0);
  int grid_x = (int)((W + BLOCK_X - 1) / BLOCK_X);
  int grid_y = (int)((H + BLOCK_Y - 1) / BLOCK_Y);
  int num_tiles = grid_x * grid_y;

  auto pre = preprocess_gaussians(
      means3D, scales, rotations, opacities, sh,
      viewmatrix, projmatrix, campos,
      W, H, tan_fovx, tan_fovy, sh_degree, scale_modifier, activate_params, sh_rest);

  auto means2d = pre[0];
  auto depths = pre[1];
  auto conic_opacity = pre[2];
  auto rgb = pre[3];
  auto radii = pre[6];
  auto cov3Ds = pre[8];
  auto clamped = pre[9];
  auto tile_bounds = pre[10];

  auto device = means3D.device();
  auto opts_i32 = means3D.options().dtype(torch::kInt32);
  auto opts_i64 = means3D.options().dtype(torch::kInt64);
  auto opts_u8 = means3D.options().dtype(torch::kUInt8);

  auto bg = bg_color.to(device).to(torch::kFloat32).contiguous();
  if (bg.numel() == 1) bg = bg.expand({3}).contiguous();

  auto& s = scratch();
  ensure_P(s, P, opts_i32, opts_u8);
  s.tiles_touched.narrow(0, 0, P).zero_();
  {
    int threads = 256;
    int blocks = (P + threads - 1) / threads;
    tiles_touched_from_bounds_kernel<<<blocks, threads>>>(
        reinterpret_cast<const int4*>(tile_bounds.data_ptr<int32_t>()),
        radii.data_ptr<int32_t>(),
        s.tiles_touched.data_ptr<int32_t>(),
        P);
  }

  cub::DeviceScan::InclusiveSum(
      s.scan_workspace.data_ptr<uint8_t>(), s.scan_bytes,
      s.tiles_touched.data_ptr<int32_t>(),
      s.point_offsets.data_ptr<int32_t>(),
      P);

  int num_rendered = 0;
  if (P > 0) {
    cudaMemcpy(&num_rendered, s.point_offsets.data_ptr<int32_t>() + P - 1,
               sizeof(int), cudaMemcpyDeviceToHost);
  }

  auto empty_ranges = torch::zeros({num_tiles, 2}, opts_i32);
  auto empty_list = torch::zeros({0}, opts_i32);
  if (num_rendered <= 0) {
    auto out = bg.view({3, 1, 1}).expand({3, (int)H, (int)W}).contiguous().clone();
    auto final_T = torch::ones({(int)H * (int)W}, means3D.options().dtype(torch::kFloat32));
    auto n_contrib = torch::zeros({(int)H * (int)W}, opts_i32);
    auto max_contrib = torch::zeros({num_tiles}, opts_i32);
    auto pixel_colors = torch::zeros({3, (int)H, (int)W}, means3D.options());
    auto bucket_off = torch::zeros({num_tiles}, opts_i32);
    auto bucket_to_tile = torch::zeros({1}, opts_i32);
    auto sampled_T = torch::zeros({1}, means3D.options().dtype(torch::kFloat16));
    auto sampled_ar = torch::zeros({1}, means3D.options().dtype(torch::kFloat16));
    auto num_buckets_t = torch::tensor({(int64_t)0}, opts_i64);
    return {out, radii, means2d, conic_opacity, rgb, cov3Ds, clamped,
            empty_ranges, empty_list, final_T, n_contrib, max_contrib, pixel_colors,
            bucket_off, bucket_to_tile, sampled_T, sampled_ar, num_buckets_t, bg};
  }

  ensure_L(s, num_rendered, opts_i64, opts_i32, opts_u8);
  ensure_tiles(s, num_tiles, opts_i32);

  {
    int threads = 256;
    int blocks = (P + threads - 1) / threads;
    duplicate_with_keys_kernel<<<blocks, threads>>>(
        P,
        depths.data_ptr<float>(),
        reinterpret_cast<const int4*>(tile_bounds.data_ptr<int32_t>()),
        s.point_offsets.data_ptr<int32_t>(),
        radii.data_ptr<int32_t>(),
        reinterpret_cast<uint64_t*>(s.keys_unsorted.data_ptr<int64_t>()),
        reinterpret_cast<uint32_t*>(s.vals_unsorted.data_ptr<int32_t>()),
        grid_x, grid_y);
  }

  int bit = (int)getHigherMsbHost((uint32_t)num_tiles);
  cub::DeviceRadixSort::SortPairs(
      s.sort_workspace.data_ptr<uint8_t>(), s.sort_bytes,
      reinterpret_cast<uint64_t*>(s.keys_unsorted.data_ptr<int64_t>()),
      reinterpret_cast<uint64_t*>(s.keys_sorted.data_ptr<int64_t>()),
      reinterpret_cast<uint32_t*>(s.vals_unsorted.data_ptr<int32_t>()),
      reinterpret_cast<uint32_t*>(s.vals_sorted.data_ptr<int32_t>()),
      num_rendered, 0, DEPTH_SORT_BITS + bit);

  s.ranges.narrow(0, 0, num_tiles).zero_();
  {
    int threads = 256;
    int blocks = (num_rendered + threads - 1) / threads;
    identify_tile_ranges_kernel<<<blocks, threads>>>(
        num_rendered,
        reinterpret_cast<const uint64_t*>(s.keys_sorted.data_ptr<int64_t>()),
        reinterpret_cast<uint2*>(s.ranges.data_ptr<int32_t>()));
  }

  // Clone so global scratch can be reused safely across backward.
  auto ranges = s.ranges.narrow(0, 0, num_tiles).clone();
  auto point_list = s.vals_sorted.narrow(0, 0, num_rendered).clone();

  auto rendered = render_sorted_list_train(
      ranges, point_list, means2d, conic_opacity, rgb, bg, W, H, grid_x, grid_y);
  // out, final_T, n_contrib, max_contrib, pixel_colors, bucket_off,
  // bucket_to_tile, sampled_T, sampled_ar, num_buckets
  return {rendered[0], radii, means2d, conic_opacity, rgb, cov3Ds, clamped,
          ranges, point_list, rendered[1], rendered[2], rendered[3], rendered[4],
          rendered[5], rendered[6], rendered[7], rendered[8], rendered[9], bg};
}
