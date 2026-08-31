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

#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("hts_stats_reset", &htsStatsReset);
  m.def("hts_stats_st_kept", &htsStatsStKept);
  m.def("hts_stats_st_cand", &htsStatsStCand);
  m.def("hts_stats_rect_tiles", &htsStatsRectTiles);
  m.def("hts_stats_aabb_tiles", &htsStatsAabbTiles);
  m.def("hts_stats_hts_tiles", &htsStatsHtsTiles);
  m.def("hts_stats_st2_cand", &htsStatsSt2Cand);
  m.def("hts_stats_st2_kept", &htsStatsSt2Kept);
  m.def("hts_stats_tie_count", &htsStatsTieCount);
  m.def("hts_stats_num_rendered", &htsStatsNumRendered);
  m.def("hts_profile_binning_ms", &htsProfileBinningMs);
  m.def("hts_profile_radix_sort_ms", &htsProfileRadixSortMs);
  m.def("hts_profile_ranges_ms", &htsProfileRangesMs);
  m.def("hts_profile_render_ms", &htsProfileRenderMs);
  m.def("hts_profile_sort_temp_bytes", &htsProfileSortTempBytes);
}