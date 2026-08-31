/*
 * TFR CUDA pybind: preprocess, fused forward, train forward/backward.
 */
#include <torch/extension.h>
#include <vector>

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
    torch::Tensor sh_rest);

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
    torch::Tensor sh_rest);

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
    int64_t num_buckets);

std::vector<torch::Tensor> preprocess_backward(
    torch::Tensor means3D,
    torch::Tensor radii,
    torch::Tensor shs,
    torch::Tensor clamped,
    torch::Tensor scales,
    torch::Tensor rotations,
    torch::Tensor cov3Ds,
    torch::Tensor viewmatrix,
    torch::Tensor projmatrix,
    torch::Tensor campos,
    torch::Tensor opacities,
    torch::Tensor dL_dmean2D,
    torch::Tensor dL_dconic,
    torch::Tensor dL_dopacity,
    torch::Tensor dL_dcolor,
    int64_t W,
    int64_t H,
    double tan_fovx,
    double tan_fovy,
    int64_t sh_degree,
    double scale_modifier);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess_gaussians", &preprocess_gaussians);
  m.def("forward_packed", &forward_packed);
  m.def("forward_packed_train", &forward_packed_train);
  m.def("render_sorted_list_backward", &render_sorted_list_backward);
  m.def("preprocess_backward", &preprocess_backward);
}
