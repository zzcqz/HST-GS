# TFR 机制原理与 HST-GS 集成说明

> **TFR = Fixed-Tile Splatting**（固定 16×16 tile 的 3DGS 式 binning + raster）  
> 路径：`HST-GS/submodules/tfr`  
> 配套文档：`[vs_vanilla_3dgs.md](vs_vanilla_3dgs.md)`（英文审稿版）、`[architecture.md](architecture.md)`（自适应 tile 研究架构）

---

## 1. 定位：TFR 是什么、不是什么

### 1.1 在 HST-GS 方法栈中的角色


| 模块       | 阶段          | 功能                                      |
| -------- | ----------- | --------------------------------------- |
| **HTS**  | 训练          | Super-Tile 视锥裁剪，减少无效 raster 工作量         |
| **SR**   | 训练          | Selective Refinement，稀疏 refine 迭代调度     |
|          |             |                                         |
| **TFR** | **评测 / 推理** | 对已训练 checkpoint 做 **forward-only 渲染加速** |


```
HST-GS 训练：HTS + SR + ACW densify（始终 native rasterizer）
HST-GS 评测：可选 TFR（render.py --use_tfr）
```



### 1.2 明确边界

**TFR 会做的：**

- 在 **eval / render** 阶段替换 raster 后端，加速 forward
- 保持与 native 渲染 **相同的图像形成模型**（16×16 tile + 深度排序 + α 混合）
- 输出与 native 渲染图 **PSNR ≈ 98–99 dB**（近乎逐像素一致）

**TFR 不会做的：**

- **不参与训练**（`train.py` 检测到 TFR 开关会直接退出）
- **不修改 densify**（与 FastGS / vanilla 3DGS densify 正交）
- **不改变高斯参数**（只读 checkpoint PLY）
- **不是 HTS / SR**（`experiments/hts_analysis/` 中的 `sr_skip` 等实验与 TFR 无关）

---



## 2. 核心原理



### 2.1 共享算法骨架（与 native 一致）

TFR 与 HST-GS 默认 `diff-gaussian-rasterization` 在**语义上**走同一条 splatting 管线：

```
高斯 (xyz, scale, rot, opacity, SH) + 相机
        │
        ▼
① Preprocess (CUDA)
   · 投影 → means2D, depths
   · 3D 协方差 → 2D conic_opacity
   · SH → RGB
   · radii / visibility / tile_bounds
        │
        ▼
② Tile binning
   · tiles_touched（由 tile_bounds 缓存推导）
   · CUB DeviceScan::InclusiveSum → point_offsets
   · duplicateWithKeys：key = (tile_id << 24) | depth_bits
   · CUB DeviceRadixSort::SortPairs
   · identifyTileRanges → ranges[tile] = [begin, end)
        │
        ▼
③ Per-tile compositing
   · 每个 16×16 tile 一个 CUDA block
   · 按 sorted point_list 从前到后 α 混合
   · 输出 RGB（+ 背景色）
```

**Takeaway：** TFR 不是新的 splatting 公式，也不是另一种 tile 划分；它复用同一套 tile + 深度排序 + 合成，保证与 native 画质可对齐。

### 2.2 三条加速杠杆（系统层特化）

加速来自 **eval 路径专用化**，而非改变图像形成数学：


| 杠杆               | Native 3DGS rasterizer                                    | TFR eval                               |
| ---------------- | --------------------------------------------------------- | --------------------------------------- |
| **Autograd 脚手架** | `_RasterizeGaussians` + `screenspace_points (P×4)`        | 单次 `forward_packed`，无 Autograd Function |
| **训练辅助缓冲**       | `geomBuffer` / `sampleBuffer` / `final_T` / `n_contrib` 等 | 默认 color-only，`with_aux=False`          |
| **属性取数**         | 训练路径写 sample-bucket；部分路径有 gather_pack                     | `point_list[i] → SoA[gid]` 直接索引         |


Native 路径即使在 `torch.no_grad()` 下评测，仍走**训练一体化** rasterizer，携带 densify / backward 的历史包袱。TFR 把 eval 做成**融合、无梯度、无 sample-bucket 的专用 forward**。

### 2.3 融合入口 `forward_packed`

核心 CUDA 融合在 `cuda/csrc/tfr_binning.cu`：

1. `preprocess_gaussians` — 投影、协方差、SH、tile_bounds
2. `tiles_touched_from_bounds` — 由缓存 bounds 统计 tile 覆盖
3. CUB InclusiveSum → `duplicate_with_keys` → CUB RadixSort → `identify_tile_ranges`
4. `render_sorted_list` — 按 sorted list 合成 RGB

对应实现文件：


| 组件                   | 路径                             |
| -------------------- | ------------------------------ |
| Preprocess           | `cuda/csrc/tfr_preprocess.cu` |
| Binning + 融合 forward | `cuda/csrc/tfr_binning.cu`    |
| Raster               | `cuda/csrc/tfr_render.cu`     |
| Backward（研究用）        | `cuda/csrc/tfr_backward.cu`   |
| Python 入口            | `python/tfr/render.py`        |




### 2.4 Raster：SoA 直接取属性

`tfr_render.cu` 中，sorted list 的每个 Gaussian id 直接从 SoA 缓冲读取：

```cuda
int gid = point_list[range.x + progress];
collected_xy[...] = means2d[2*gid], means2d[2*gid+1];
collected_conic_opacity[...] = conic_opacity[gid];
collected_rgb[...] = rgb[3*gid], rgb[3*gid+1], rgb[3*gid+2];
```

α 混合规则与 Kerbl 原版一致（`power`、`alpha = min(0.99, opacity*exp(power))`、`T` 累积、early stop）。

Eval 路径使用 raw 参数 + CUDA 内激活（`activate_params=True`），避免 Python 侧重复激活开销。

### 2.5 产品版 vs 研究版

**当前 HST-GS 正式接入：Fixed-Tile（16×16）**

- tile 网格与 native 3DGS 相同
- 加速 purely from systems overhead reduction

**研究版（未接入默认** `--use_tfr` **路径）**

见 `[architecture.md](architecture.md)`：

- 屏幕空间复杂度估计 C(b)
- 自适应 tile merge/split（16 / 32 / 64）
- Complexity-aware assignment + GatherPack

该 adaptive 设计存在于独立 benchmark / 文档中，**不改变** HST-GS 默认 eval 行为。

---



## 3. HST-GS 代码改动清单



### 3.1 主仓库集成层（量小但关键）


| 文件                              | 改动                                                     |
| ------------------------------- | ------------------------------------------------------ |
| `arguments/__init__.py`         | `PipelineParams.use_tfr = False`                      |
| `gaussian_renderer/__init__.py` | TFR / native 路由；训练态禁止 TFR                            |
| `train.py`                      | 启动 guard：`--use_tfr` 或 `HSTGS_USE_TFR=1` → `exit(2)` |
| `render.py`                     | 写 `render_fps_tfr.json` / `render_fps_native.json`    |
| `submodules/TFR.md`            | 构建与用法简述                                                |


**未改动：** `scene/gaussian_model.py`、HTS / SR 调度、densify 逻辑、`utils/`*。

### 3.2 渲染路由逻辑

```
render(view, pc, pipe, bg)
    │
    ├─ pipe.use_tfr 或 HSTGS_USE_TFR=1 ?
    │       YES → _render_with_tfr()
    │              · 检查 torch.is_grad_enabled() → 报错
    │              · import tfr.render
    │              · with_aux=False
    │
    └─ NO  → native GaussianRasterizer
              · screenspace_points (P×4, requires_grad=True)
              · 完整 Autograd + sample buckets
```



### 3.3 新增 submodule `submodules/tfr/`

```
submodules/tfr/
├── python/tfr/
│   ├── render.py          # HST-GS 兼容 render()
│   ├── hstgs_bridge.py    # 独立 benchmark 加载 checkpoint
│   └── metrics.py
├── cuda/csrc/
│   ├── tfr_preprocess.cu
│   ├── tfr_binning.cu    # forward_packed
│   ├── tfr_render.cu
│   ├── tfr_backward.cu   # Phase 1/2 研究
│   └── tfr_ext.cu
├── configs/default.yaml
└── docs/
    ├── vs_vanilla_3dgs.md
    ├── architecture.md
    └── TFR_MECHANISM_HSTGS.md   # 本文档
```



### 3.4 实验脚本

`experiments/full_matrix/run_*.sh` 典型三阶段：

1. `train.py` — native raster + HTS / SR
2. `render.py --use_tfr` — TFR FPS / metrics
3. `render.py`（native）— 最终 `results.json`

---



## 4. 与 Vanilla / FastGS / HTS 的对比


| 维度       | Vanilla 3DGS | FastGS        | HST-GS (HTS+SR)   | TFR                         |
| -------- | ------------ | ------------- | ----------------- | ---------------------------- |
| 改动对象     | —            | 训练 densify 策略 | 训练 raster 裁剪 + SR | **仅 eval raster**            |
| 是否改高斯    | 是            | 是             | 是                 | **否**                        |
| Autograd | 是            | 是             | 是                 | **否（eval）**                  |
| Tile 网格  | 16×16        | 16×16         | 16×16             | 16×16（fixed）                 |
| 典型加速     | baseline     | 训练配方差异        | 训练 ~5–6×          | 渲染 ~2.3×                     |
| 画质参照     | vs GT        | vs GT         | vs GT             | vs **native render** ≈ 99 dB |


```
┌─────────────────────────────────────────────────────────┐
│ 训练 (HST-GS)                                            │
│   train.py → HTS + SR + ACW densify                     │
│            → native diff-gaussian-rasterization          │
│            → screenspace_points 供 densify 梯度          │
└───────────────────────────┬─────────────────────────────┘
                            │ iteration_30000 PLY
                            ▼
┌─────────────────────────────────────────────────────────┐
│ 评测 (可选 TFR)                                         │
│   render.py --use_tfr                                   │
│     → tfr_cuda.forward_packed                           │
│     → 无 Autograd / 无 sample buckets / SoA 直取         │
└─────────────────────────────────────────────────────────┘
```

---



## 5. 参数与用法



### 5.1 用户可见开关


| 参数               | 默认      | 说明                           |
| ---------------- | ------- | ---------------------------- |
| `--use_tfr`     | `False` | 仅 `render.py`                |
| `HSTGS_USE_TFR` | 未设置     | 环境变量，`1` / `true` / `yes` 启用 |
| `TFR_ROOT`      | 自动      | 覆盖 submodule 路径              |




### 5.2 TFR 不支持的 pipe 选项

以下选项在 TFR 路径下会 `NotImplementedError`：

- `separate_sh`
- `override_color`
- `convert_SHs_python`
- `compute_cov3D_python`



### 5.3 内部默认（`configs/default.yaml`）


| 参数                      | 默认    | 说明                  |
| ----------------------- | ----- | ------------------- |
| `base_tile`             | 16    | 固定 tile 大小          |
| `pack_layout`           | `soa` | Structure-of-arrays |
| `stable_depth_tiebreak` | `id`  | 深度排序 tie-break      |




### 5.4 构建与调用

```bash
# 构建 CUDA 扩展（GPU 节点）
cd HST-GS/submodules/tfr/cuda
python setup.py build_ext --inplace

# Eval 渲染
cd HST-GS
python render.py -m <model_path> --use_tfr --skip_train
# 输出：render_fps_tfr.json

# 或通过环境变量
HSTGS_USE_TFR=1 python render.py -m <model_path> --skip_train
```



### 5.5 训练禁止

```bash
# 以下会报错退出（exit code 2）
python train.py ... --use_tfr
HSTGS_USE_TFR=1 python train.py ...
```

---



## 6. 定量结果（eval，iteration_30000 PLY）

协议：`scripts/bench_tfr_render.py --all-13`，每场景全部 test 相机，warmup=2，repeats=5。


| 指标             | 典型值          |
| -------------- | ------------ |
| Native FPS（均值） | ~175         |
| TFR FPS（均值）   | ~407         |
| Speedup        | **~2.32×**   |
| PSNR vs native | **~98.7 dB** |


分场景详见 `vs_vanilla_3dgs.md` [§7](vs_vanilla_3dgs.md)。原始 JSON：`outputs/bench_render/bench_render_13.json`。

---



## 7. 论文 / 报告撰写建议

1. **TFR 是 eval accelerator**，不要表述为「训练 rasterizer 替换」或「densify 改进」。
2. **Fixed-Tile 是当前接入版本**；adaptive tile 研究单独叙述，不与默认 `--use_tfr` 混为一谈。
3. 主指标：**FPS speedup + PSNR vs native**（~99 dB），而非仅 vs GT。
4. `experiments/hts_analysis/RESULT_REPORT.md` 聚焦 HTS / SR / depth，**不含 TFR**；TFR 定量见本文 §6 与 `vs_vanilla_3dgs.md`。



### Method 段落模板（英文）

> **TFR (eval).** We keep the standard 3DGS 16×16 tile rasterization pipeline (preprocess, CUB key duplication and radix sort, per-tile front-to-back α-blending) but specialize the **evaluation** path: a single fused CUDA forward without Autograd scaffolding, without training sample-bucket traffic, and with direct SoA attribute fetch by sorted Gaussian index. On Kerbl-style checkpoints at 30k iterations, TFR improves end-to-end validation FPS by approximately **2.3×** on average across 13 scenes while matching the native renderer to **~99 dB** PSNR, confirming that the speedup comes from systems specialization rather than a change of the image formation model.

---



## 8. 相关文档索引


| 文档                                           | 内容                        |
| -------------------------------------------- | ------------------------- |
| `[../TFR.md](../../TFR.md)`                | 构建与用法速查                   |
| `[vs_vanilla_3dgs.md](vs_vanilla_3dgs.md)`   | 与 vanilla 差异、审稿版说明、13 场景表 |
| `[architecture.md](architecture.md)`         | 自适应 tile 研究架构             |
| `[experiments.md](experiments.md)`           | Bench 协议与指标定义             |
| `[full_13_analysis.md](full_13_analysis.md)` | Fixed vs adaptive tile 分析 |


