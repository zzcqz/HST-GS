# TFR Architecture

## Goal

Accelerate 3D Gaussian rasterization by (1) adapting tile granularity to **screen-space complexity**, (2) assigning Gaussians under load constraints, and (3) **repacking** per-tile attribute lists into contiguous GPU memory.

Decoupled from HST-GS (HTS / SR).

## Pipeline

```
Gaussians (xyz, cov, SH, opacity, ...)
      │
      ▼
Screen-space Complexity Estimator
      │
      ▼
Adaptive Tile Generator          # Contribution 1
      │
      ▼
Visibility Scheduler             # Contribution 1 (priority)
      │
      ▼
Complexity-aware Tile Assignment # Contribution 2
      │
      ▼
Memory-aware Rasterizer          # Contribution 3 (GatherPack + raster)
      │
      ▼
Image
```

## Contribution 1 — Adaptive Tile Scheduler

Base tile \(B = 16\times16\). Merge levels \(\{16,32,64\}\) (\(k\in\{0,1,2\}\)).

Outputs `AdaptiveTile{rect, level, complexity, priority, active}`.

### Screen-space complexity

For base tile \(b\):

\[
C(b)=\sum_{g\in\mathcal{N}(b)}\big(w_\alpha\cdot\hat\alpha_g + w_a\cdot A_g(b)\big)
\]

Default \(w_\alpha=w_a=1\). For merge candidate \(T=\bigcup b_i\): \(C(T)=\sum C(b_i)\).

Merge if **density** \(C(T)/\#bases \le C_{\mathrm{merge}}\) and no child is `must_split`
(absolute thresholds failed on playroom where mean \(C(b)\approx 2300\)).
Priority \(P(T)=C(T)\); tiles with \(C=0\) are inactive.

Priority key: `quantize(C) << 16 | morton(T)` (high complexity first; spatial locality on ties).

Complexity 图可 `scripts/vis_complexity.py` / phase outputs 导出热力图。

## Contribution 2 — Complexity-aware Tile Assignment

- Gaussians attach only to **active leaf** adaptive tiles.
- If \(C(t) > C_{\mathrm{split}}\), force level-0 tiles and optionally split work into waves.
- Sort key: `adaptive_tile_id << 32 | depth_key` (depth + stable id).

## Contribution 3 — Memory-aware Rasterization

```
AdaptiveTile t
  └── PackedGaussianList[t]  # contiguous SoA/AOS in arena
        means2d | depth | conic_opacity | rgb | ...
```

`GatherPack` copies global SoA → `packed_pool` indexed by `tile_offset[t], tile_count[t]`.  
Raster kernel streams contiguous slots (coalesced loads).

v0: forward correctness vs fixed-tile golden path. Backward scatter deferred to v1.

## Data structures

```text
struct AdaptiveTile {
  uint16_t x0,y0,x1,y1;   // pixel rect
  uint8_t  level;         // 0=16,1=32,2=64
  uint8_t  flags;         // active / must_split
  float    complexity;
  uint32_t priority_key;
};

struct TileAssignment {
  uint32_t tile_id;
  uint32_t list_begin, list_count;
};

struct PackedSlab {
  float2* means2d; float* depths; float4* conic_opacity; float3* rgbs;
  uint32_t* tile_offset; uint32_t* tile_count;
  uint32_t* gauss_ids;    // optional back-map for backward
};
```

## Prototype stages

| Stage | Content |
|-------|---------|
| v0 | Python complexity + merge + pack; fixed-tile golden reference |
| v0.1 | CUDA complexity reduce + adaptive ids |
| v0.2 | Adaptive duplicate/sort/ranges |
| v0.3 | GatherPack + memory-aware kernel |
| v1 | Differentiable scatter backward |

## Defaults (`configs/default.yaml`)

- `base_tile: 16`
- `merge_levels: [16, 32, 64]`
- `C_merge: 200.0`
- `C_split: 400.0`
- `w_alpha: 1.0`, `w_area: 1.0`
