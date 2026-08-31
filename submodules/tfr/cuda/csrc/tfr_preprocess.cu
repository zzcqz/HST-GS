/*
 * Screen-space preprocess matching golden / 3DGS CUDA (geometry + SH → RGB).
 * Outputs per-Gaussian means2d, depths, conic_opacity, rgb, anisotropic extents.
 */
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cmath>
#include <vector>

#include <glm/glm.hpp>

namespace cg = cooperative_groups;

namespace {

__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
    1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
    -1.0925484305920792f, 0.5462742152960396f};
__device__ const float SH_C3[] = {
    -0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f,
    0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f,
    -0.5900435899266435f};

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
  return {
      matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
  };
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
  return {
      matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
      matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15],
  };
}

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
  return ((v + 1.0f) * S - 1.0f) * 0.5f;
}

__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
  glm::mat3 S = glm::mat3(1.0f);
  S[0][0] = mod * scale.x;
  S[1][1] = mod * scale.y;
  S[2][2] = mod * scale.z;
  glm::vec4 q = rot;
  float r = q.x, x = q.y, y = q.z, z = q.w;
  glm::mat3 R = glm::mat3(
      1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
      2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
      2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));
  glm::mat3 M = S * R;
  glm::mat3 Sigma = glm::transpose(M) * M;
  cov3D[0] = Sigma[0][0];
  cov3D[1] = Sigma[0][1];
  cov3D[2] = Sigma[0][2];
  cov3D[3] = Sigma[1][1];
  cov3D[4] = Sigma[1][2];
  cov3D[5] = Sigma[2][2];
}

__device__ float3 computeCov2D(
    const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy,
    const float* cov3D, const float* viewmatrix)
{
  float3 t = transformPoint4x3(mean, viewmatrix);
  const float limx = 1.3f * tan_fovx;
  const float limy = 1.3f * tan_fovy;
  const float txtz = t.x / t.z;
  const float tytz = t.y / t.z;
  t.x = min(limx, max(-limx, txtz)) * t.z;
  t.y = min(limy, max(-limy, tytz)) * t.z;

  glm::mat3 J = glm::mat3(
      focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
      0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
      0, 0, 0);
  glm::mat3 W = glm::mat3(
      viewmatrix[0], viewmatrix[4], viewmatrix[8],
      viewmatrix[1], viewmatrix[5], viewmatrix[9],
      viewmatrix[2], viewmatrix[6], viewmatrix[10]);
  glm::mat3 T = W * J;
  glm::mat3 Vrk = glm::mat3(
      cov3D[0], cov3D[1], cov3D[2],
      cov3D[1], cov3D[3], cov3D[4],
      cov3D[2], cov3D[4], cov3D[5]);
  glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;
  return {float(cov[0][0]), float(cov[0][1]), float(cov[1][1])};
}

__device__ glm::vec3 sh_band(const float* sh_dc, const float* sh_rest, int idx, int rest_bands, int band)
{
  // Combined layout: sh is (P, 1+rest_bands, 3) passed as sh_dc with sh_rest==nullptr
  if (sh_rest == nullptr)
    return ((const glm::vec3*)sh_dc)[idx * (1 + rest_bands) + band];
  if (band == 0)
    return ((const glm::vec3*)sh_dc)[idx];
  return ((const glm::vec3*)sh_rest)[idx * rest_bands + (band - 1)];
}

__device__ glm::vec3 computeColorFromSH(
    int idx, int deg, int rest_bands, const glm::vec3* means, glm::vec3 campos,
    const float* sh_dc, const float* sh_rest, uint8_t* clamped)
{
  glm::vec3 pos = means[idx];
  glm::vec3 dir = pos - campos;
  dir = dir / glm::length(dir);
  glm::vec3 result = SH_C0 * sh_band(sh_dc, sh_rest, idx, rest_bands, 0);
  if (deg > 0) {
    float x = dir.x, y = dir.y, z = dir.z;
    result = result - SH_C1 * y * sh_band(sh_dc, sh_rest, idx, rest_bands, 1)
                    + SH_C1 * z * sh_band(sh_dc, sh_rest, idx, rest_bands, 2)
                    - SH_C1 * x * sh_band(sh_dc, sh_rest, idx, rest_bands, 3);
    if (deg > 1) {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;
      result = result + SH_C2[0] * xy * sh_band(sh_dc, sh_rest, idx, rest_bands, 4)
               + SH_C2[1] * yz * sh_band(sh_dc, sh_rest, idx, rest_bands, 5)
               + SH_C2[2] * (2.0f * zz - xx - yy) * sh_band(sh_dc, sh_rest, idx, rest_bands, 6)
               + SH_C2[3] * xz * sh_band(sh_dc, sh_rest, idx, rest_bands, 7)
               + SH_C2[4] * (xx - yy) * sh_band(sh_dc, sh_rest, idx, rest_bands, 8);
      if (deg > 2) {
        result = result + SH_C3[0] * y * (3 * xx - yy) * sh_band(sh_dc, sh_rest, idx, rest_bands, 9)
                 + SH_C3[1] * xy * z * sh_band(sh_dc, sh_rest, idx, rest_bands, 10)
                 + SH_C3[2] * y * (4 * zz - xx - yy) * sh_band(sh_dc, sh_rest, idx, rest_bands, 11)
                 + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh_band(sh_dc, sh_rest, idx, rest_bands, 12)
                 + SH_C3[4] * x * (4 * zz - xx - yy) * sh_band(sh_dc, sh_rest, idx, rest_bands, 13)
                 + SH_C3[5] * z * (xx - yy) * sh_band(sh_dc, sh_rest, idx, rest_bands, 14)
                 + SH_C3[6] * x * (xx - 3 * yy) * sh_band(sh_dc, sh_rest, idx, rest_bands, 15);
      }
    }
  }
  result += 0.5f;
  if (clamped) {
    clamped[3 * idx + 0] = (uint8_t)(result.x < 0.0f);
    clamped[3 * idx + 1] = (uint8_t)(result.y < 0.0f);
    clamped[3 * idx + 2] = (uint8_t)(result.z < 0.0f);
  }
  return glm::max(result, 0.0f);
}

__global__ void preprocess_kernel(
    int P, int D, int M,
    const float* __restrict__ orig_points,
    const float* __restrict__ scales,
    const float* __restrict__ rotations,
    const float* __restrict__ opacities,
    const float* __restrict__ sh_dc,
    const float* __restrict__ sh_rest,
    int rest_bands,
    const float* __restrict__ viewmatrix,
    const float* __restrict__ projmatrix,
    const float* __restrict__ cam_pos,
    int W, int H,
    float tan_fovx, float tan_fovy,
    float focal_x, float focal_y,
    float scale_modifier,
    int activate_params,
    float* __restrict__ means2d,
    float* __restrict__ depths,
    float* __restrict__ conic_opacity,
    float* __restrict__ rgb,
    float* __restrict__ extent_x,
    float* __restrict__ extent_y,
    int32_t* __restrict__ radii,
    uint8_t* __restrict__ visible,
    float* __restrict__ cov3Ds,       // nullable (P,6)
    uint8_t* __restrict__ clamped,    // nullable (P,3)
    int4* __restrict__ tile_bounds)   // nullable (P,) minx,miny,maxx,maxy
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= P) return;

  radii[idx] = 0;
  visible[idx] = 0;
  extent_x[idx] = 0;
  extent_y[idx] = 0;
  if (tile_bounds)
    tile_bounds[idx] = make_int4(0, 0, 0, 0);
  if (clamped) {
    clamped[3 * idx + 0] = 0;
    clamped[3 * idx + 1] = 0;
    clamped[3 * idx + 2] = 0;
  }

  float base_opacity = opacities[idx];
  if (activate_params)
    base_opacity = 1.0f / (1.0f + expf(-base_opacity));  // sigmoid
  if (base_opacity < 1.0f / 255.0f)
    return;

  float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};
  float3 p_view = transformPoint4x3(p_orig, viewmatrix);
  if (p_view.z <= 0.2f)
    return;

  float4 p_hom = transformPoint4x4(p_orig, projmatrix);
  float p_w = 1.0f / (p_hom.w + 1e-7f);
  float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};

  float cov3D[6];
  glm::vec3 scale = {scales[3 * idx], scales[3 * idx + 1], scales[3 * idx + 2]};
  glm::vec4 rot = {
      rotations[4 * idx], rotations[4 * idx + 1], rotations[4 * idx + 2], rotations[4 * idx + 3]};
  if (activate_params) {
    scale = {expf(scale.x), expf(scale.y), expf(scale.z)};
    float inv = rsqrtf(fmaxf(1e-12f, rot.x * rot.x + rot.y * rot.y + rot.z * rot.z + rot.w * rot.w));
    rot *= inv;
  }
  computeCov3D(scale, scale_modifier, rot, cov3D);
  if (cov3Ds) {
    for (int i = 0; i < 6; ++i)
      cov3Ds[6 * idx + i] = cov3D[i];
  }

  float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);
  cov.x += 0.3f;
  cov.z += 0.3f;
  float det = cov.x * cov.z - cov.y * cov.y;
  if (det == 0.0f)
    return;
  float det_inv = 1.0f / det;
  float3 conic = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};

  float mid = 0.5f * (cov.x + cov.z);
  float lambda1 = mid + sqrtf(fmaxf(0.1f, mid * mid - det));
  float lambda2 = mid - sqrtf(fmaxf(0.1f, mid * mid - det));
  float my_radius = ceilf(3.f * sqrtf(fmaxf(lambda1, lambda2)));
  float2 point_image = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)};

  glm::vec3 campos = {cam_pos[0], cam_pos[1], cam_pos[2]};
  glm::vec3 result = computeColorFromSH(
      idx, D, rest_bands, (const glm::vec3*)orig_points, campos, sh_dc, sh_rest, clamped);

  float alpha_log = logf(255.0f * base_opacity);
  if (alpha_log <= 0.0f)
    return;
  float q = 2.0f * alpha_log;
  float det_conic = conic.x * conic.z - conic.y * conic.y;
  if (det_conic <= 0.0f)
    return;
  float ex = sqrtf(fmaxf(0.0f, q * conic.z / det_conic));
  float ey = sqrtf(fmaxf(0.0f, q * conic.x / det_conic));

  // Rough image cull
  if (point_image.x + ex < 0 || point_image.x - ex >= W ||
      point_image.y + ey < 0 || point_image.y - ey >= H)
    return;

  means2d[2 * idx] = point_image.x;
  means2d[2 * idx + 1] = point_image.y;
  depths[idx] = p_view.z;
  conic_opacity[4 * idx] = conic.x;
  conic_opacity[4 * idx + 1] = conic.y;
  conic_opacity[4 * idx + 2] = conic.z;
  conic_opacity[4 * idx + 3] = base_opacity;
  rgb[3 * idx] = result.x;
  rgb[3 * idx + 1] = result.y;
  rgb[3 * idx + 2] = result.z;
  extent_x[idx] = ex;
  extent_y[idx] = ey;
  radii[idx] = (int)my_radius;
  visible[idx] = 1;

  // Cache the AABB tile bounds so binning can skip recomputing the extent.
  if (tile_bounds) {
    const int grid_x = (W + 15) / 16;
    const int grid_y = (H + 15) / 16;
    const int tile_min_x = max(0, (int)floorf((point_image.x - ex) / 16.0f));
    const int tile_min_y = max(0, (int)floorf((point_image.y - ey) / 16.0f));
    const int tile_max_x = min(grid_x, (int)floorf((point_image.x + ex) / 16.0f) + 1);
    const int tile_max_y = min(grid_y, (int)floorf((point_image.y + ey) / 16.0f) + 1);
    tile_bounds[idx] = make_int4(tile_min_x, tile_min_y, tile_max_x, tile_max_y);
  }
}

} // namespace

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
    torch::Tensor sh_rest)
{
  TORCH_CHECK(means3D.is_cuda() && means3D.dim() == 2 && means3D.size(1) == 3, "means3D (P,3) CUDA");
  int P = (int)means3D.size(0);
  int D = (int)sh_degree;
  const bool split_sh = sh_rest.defined() && sh_rest.numel() > 0;
  int rest_bands = 0;
  if (split_sh)
    rest_bands = (int)sh_rest.size(1);
  else
    rest_bands = sh.numel() ? std::max(0, (int)sh.size(1) - 1) : 0;

  auto opts = means3D.options().dtype(torch::kFloat32);
  auto means2d = torch::zeros({P, 2}, opts);
  auto depths = torch::zeros({P}, opts);
  auto conic_opacity = torch::zeros({P, 4}, opts);
  auto rgb = torch::zeros({P, 3}, opts);
  auto extent_x = torch::zeros({P}, opts);
  auto extent_y = torch::zeros({P}, opts);
  auto radii = torch::zeros({P}, means3D.options().dtype(torch::kInt32));
  auto visible = torch::zeros({P}, means3D.options().dtype(torch::kUInt8));
  auto cov3Ds = torch::zeros({P, 6}, opts);
  auto clamped = torch::zeros({P, 3}, means3D.options().dtype(torch::kUInt8));
  auto tile_bounds = torch::zeros({P, 4}, means3D.options().dtype(torch::kInt32));

  if (P == 0) {
    return {means2d, depths, conic_opacity, rgb, extent_x, extent_y, radii, visible, cov3Ds, clamped, tile_bounds};
  }

  float focal_x = (float)W / (2.0f * (float)tan_fovx);
  float focal_y = (float)H / (2.0f * (float)tan_fovy);

  // Ensure contiguous layouts matching golden
  auto m3 = means3D.contiguous();
  auto sc = scales.contiguous();
  auto rot = rotations.contiguous();
  auto op = opacities.contiguous().view({-1});
  auto shc = sh.contiguous();
  auto shr = split_sh ? sh_rest.contiguous() : sh;
  auto vm = viewmatrix.contiguous();
  auto pm = projmatrix.contiguous();
  auto cp = campos.contiguous().view({-1});

  int threads = 256;
  int blocks = (P + threads - 1) / threads;
  preprocess_kernel<<<blocks, threads>>>(
      P, D, rest_bands,
      m3.data_ptr<float>(),
      sc.data_ptr<float>(),
      rot.data_ptr<float>(),
      op.data_ptr<float>(),
      shc.data_ptr<float>(),
      split_sh ? shr.data_ptr<float>() : nullptr,
      rest_bands,
      vm.data_ptr<float>(),
      pm.data_ptr<float>(),
      cp.data_ptr<float>(),
      (int)W, (int)H,
      (float)tan_fovx, (float)tan_fovy,
      focal_x, focal_y,
      (float)scale_modifier,
      (int)activate_params,
      means2d.data_ptr<float>(),
      depths.data_ptr<float>(),
      conic_opacity.data_ptr<float>(),
      rgb.data_ptr<float>(),
      extent_x.data_ptr<float>(),
      extent_y.data_ptr<float>(),
      radii.data_ptr<int32_t>(),
      visible.data_ptr<uint8_t>(),
      cov3Ds.data_ptr<float>(),
      clamped.data_ptr<uint8_t>(),
      reinterpret_cast<int4*>(tile_bounds.data_ptr<int32_t>()));

  return {means2d, depths, conic_opacity, rgb, extent_x, extent_y, radii, visible, cov3Ds, clamped, tile_bounds};
}
