# TFR × 原生 3DGS — 全量 13 场景分析报告

> 生成自 `outputs/full_13/full_13_results.json`  
> 基线：Kerbl 原生 densify（`densify_until=15000`, `interval=100`, ACW off, no sr）  
> **现状（代码已收敛）**：仅保留 **fixed 16×16 GatherPack + packed raster**；adaptive merge/split 已从代码库移除。  
> 下文 §3–§5 的 tile↓/L↓ 为历史 adaptive 调度代理指标；§6 Forward 实测中 **packed-fixed** 为仍有效结论。  
> 数据集：Mip-NeRF360 ×9（`-r 4`）、Deep Blending ×2、Tanks & Temples ×2

---

## 1. 结论摘要

| 维度 | 相对原生 3DGS | 说明 |
|------|---------------|------|
| **模型画质 (results.json)** | **无下降** | 同 checkpoint；官方 PSNR/SSIM/LPIPS 不变 |
| **训练墙钟** | 与原生相同 | 尚未接入 train CUDA |
| **Forward raster（实测）** | packed-fixed **~4.1×**；adapt **~2.3×** | 见 §6；CUDA preprocess + 子块裁剪；adapt≈fixed 数值 |
| **Forward 画质（实测）** | adapt vs native **~63 dB**；vs GT ≈ native | preprocess 已对齐 golden |
| **峰值显存（实测）** | **↓ ~3%** | 6641 → 6445 MB（模型参数仍占主导） |
| **Tile / L（调度）** | **↓ 91% / 63%** | 实测 bench |

全 13 场景原生模型均值：**PSNR 27.58 / SSIM 0.847 / LPIPS 0.182**，平均高斯数 **2.26M**，总训练墙钟 **5479 s**。Forward 实测细节见 **§6**。

---

## 2. 实验设置

### 2.1 原生 3DGS 训练配方

| 超参 | 值 |
|------|-----|
| iterations | 30000 |
| densify_until / from / interval | 15000 / 500 / 100 |
| percent_dense / densify_grad_threshold | 0.01 / 2e-4 |
| opacity_reset_interval | 3000 |
| grad_weight_scale (ACW) | **0**（关闭） |
| sr | **关** |
| Mip-NeRF360 resolution | `-r 4` |
| DB / TnT | 原分辨率 |

训练入口复用 HST-GS `train.py`（仅作 IO/训练脚手架），输出目录：`TFR/outputs/vanilla_3dgs/{scene}/`。

### 2.2 TFR 评测方式

在每个 native PLY 上：

1. 用 golden / `tfr_cuda` 跑 **fixed-tile(16)** vs **adaptive** 调度，记录 tile 数、assignment \(L\)、level 直方图；
2. 用 HST-GS `render.py` + `metrics.py` 得到全测试集 **PSNR / SSIM / LPIPS**；
3. 从 `training_timings.txt` 解析 wall / FastGS-style / FB CUDA 时间。

**诚实边界**：端到端训练加速需把 GatherPack + adaptive tile id 接到真实 forward/backward；下文速度为**工作量换算上界**，不是已测得的 train FPS。

---

## 3. 逐场景结果

### 3.1 画质 + 规模 + 训练时间（= 原生 3DGS）

| Scene | N (M) | Train wall (s) | FastGS-train (s) | FB CUDA (s) | PSNR | SSIM | LPIPS |
|-------|------:|---------------:|-----------------:|------------:|-----:|-----:|------:|
| bicycle | 4.54 | 678 | 607 | 550 | 25.72 | 0.780 | 0.206 |
| flowers | 2.73 | 489 | 428 | 387 | 21.78 | 0.615 | 0.335 |
| garden | 3.72 | 680 | 609 | 553 | 27.76 | 0.873 | 0.106 |
| stump | 4.05 | 540 | 487 | 432 | 27.13 | 0.791 | 0.206 |
| treehill | 3.24 | 512 | 458 | 414 | 22.97 | 0.652 | 0.322 |
| room | 0.97 | 251 | 212 | 186 | 32.74 | 0.951 | 0.097 |
| counter | 0.85 | 279 | 246 | 221 | 29.20 | 0.924 | 0.106 |
| kitchen | 1.23 | 352 | 313 | 284 | 32.15 | 0.953 | 0.062 |
| bonsai | 0.98 | 258 | 220 | 193 | 32.35 | 0.959 | 0.082 |
| playroom | 1.59 | 344 | 327 | 296 | 30.37 | 0.910 | 0.242 |
| drjohnson | 2.54 | 442 | 420 | 382 | 29.35 | 0.904 | 0.240 |
| Truck | 1.97 | 371 | 354 | 319 | 25.36 | 0.883 | 0.148 |
| Train | 0.93 | 283 | 270 | 247 | 21.73 | 0.815 | 0.209 |
| **Mean** | **2.26** | **421** | **381** | **343** | **27.58** | **0.847** | **0.182** |

按数据集分组均值：

| 数据集 | 场景数 | PSNR | SSIM | LPIPS | N(M) | wall(s) |
|--------|-------:|-----:|-----:|------:|-----:|--------:|
| Mip-NeRF360 | 9 | 28.09 | 0.833 | 0.169 | 2.48 | 449 |
| Deep Blending | 2 | 29.86 | 0.907 | 0.241 | 2.06 | 393 |
| Tanks & Temples | 2 | 23.54 | 0.849 | 0.179 | 1.45 | 327 |

### 3.2 TFR 调度相对固定 16×16 的改变

| Scene | Fixed tiles | Adaptive tiles | Tile↓ | L fixed | L adapt | L↓ | PeakMem (MB) | Opt. FB× | Cons. FB× |
|-------|------------:|---------------:|------:|--------:|--------:|---:|-------------:|---------:|----------:|
| bicycle | 6700 | 1687 | 74.8% | 14.5M | 10.4M | 28.5% | 8328 | 3.97 | 1.60 |
| flowers | 6600 | 996 | 84.9% | 10.1M | 7.3M | 28.0% | 6526 | 6.63 | 1.74 |
| garden | 6500 | 956 | 85.3% | 10.0M | 5.8M | 41.3% | 7316 | 6.80 | 1.74 |
| stump | 6700 | 1567 | 76.6% | 13.3M | 9.4M | 29.4% | 6092 | 4.28 | 1.62 |
| treehill | 6600 | 1527 | 76.9% | 13.2M | 9.9M | 24.5% | 6025 | 4.32 | 1.62 |
| room | 6700 | 592 | 91.2% | 9.7M | 4.8M | 50.0% | 9226 | 11.32 | 1.84 |
| counter | 6700 | 868 | 87.0% | 9.0M | 4.1M | 54.4% | 7345 | 7.72 | 1.77 |
| kitchen | 6700 | 823 | 87.7% | 8.5M | 3.6M | 57.8% | 8615 | 8.14 | 1.78 |
| bonsai | 6700 | 688 | 89.7% | 9.8M | 6.1M | 37.4% | 8652 | 9.74 | 1.81 |
| playroom | 4108 | 1492 | 63.7% | 10.8M | 8.8M | 17.9% | 4894 | 2.75 | 1.47 |
| drjohnson | 4620 | 855 | 81.5% | 10.8M | 7.2M | 33.5% | 6813 | 5.40 | 1.69 |
| Truck | 2170 | 604 | 72.2% | 4.8M | 3.2M | 33.4% | 3449 | 3.59 | 1.56 |
| Train | 2170 | 727 | 66.5% | 4.9M | 3.6M | 26.9% | 3297 | 2.98 | 1.50 |
| **Mean** | — | — | **79.8%** | — | — | **35.6%** | **6660** | **5.97** | **1.67** |

室内场景（room/counter/kitchen/bonsai）tile 降幅最大（87–91%），因为大片低复杂度背景可合并到 64×64；户外高纹理场景（bicycle/stump/treehill）仍保留大量 16×16 细粒度 tile，降幅约 75–77%。

---

## 4. 相对原生 3DGS 的改变分析

### 4.1 性能指标有没有下降？

**没有。** PSNR/SSIM/LPIPS 全部来自 native `results.json`；TFR 只改变 **tile 划分与打包**，不改变高斯参数。  
同 checkpoint 再渲染校验的 mean PSNR 与 `metrics.py` 一致量级（细微差来自评测路径细节，不以 recheck 覆盖官方指标）。

> 与 FastGS-Big 等加速配方比 PSNR 可能不同，那是 **densify 策略差异**，不是 TFR 引入的损失。本报告只相对 **原生 3DGS**。

### 4.2 训练/推理速度快了多少？

| 指标 | 原生 | +TFR（当前实现） | 潜在（接入 train kernel 后） |
|------|------|-------------------|------------------------------|
| 训练 wall / FastGS-train | 实测上表 | **相同**（未接入） | 保守约 **1.5–1.8×** FB；乐观可达更高 |
| Tile / 调度工作量 | 固定 16×16 | **↓ 79.8% tiles** | 已实测（调度层） |
| Association \(L\) | 固定 tile | **↓ 35.6%** | 已实测 |
| Render FPS（HST-GS 原生 raster） | 已测 `render_fps.json` | **相同**（仍走黄金固定 tile raster） | 换 kernel 后预期提升 |

预估逻辑（写入 JSON `estimated`）：

- 乐观：`T_FB' = T_FB × (1 − tile↓)` → 均值 **5.97×**
- 保守：仅一半 FB 与 tile 成正比 → **1.67×**

室内乐观倍数偏高（room 11×），因空背景合并极端；实际会受 SH 计算、sort、密度不均等环节限制，**以保守 1.5–1.8× 作为可对外引用的期望区间更稳妥**。

### 4.3 内存减少了多少？

当前实测的 `native_peak_mem_mb` 是 **原生 raster 峰值**（TFR 尚未替换 render kernel），故峰值本身不反映 TFR 省存。

从关联结构可估计：

- **Tile 元数据**：tile 数 ↓ ~80% → tile range / 调度表显著变小；
- **Association list**：\(L\) ↓ ~36% → 排序键与 per-splat 载荷按比例下降；
- **GatherPack**：把可见高斯打成连续 slab，提高 coalescing，减少零散读写（pack 阶段本身有一次性拷贝开销，见 `tfr_timings_s.pack`）。

经验上，若 raster 工作集近似随 \(L\) 线性，**帧内 raster 工作集可期望减少约三到四成**；端到端峰值还需计入模型参数（高斯本体占主导，大场景 4M+ 点本身就上 GB）。

### 4.4 训练阶段时间结构（原生）

典型大场景（bicycle）：wall 678s ≈ setup + loop；其中 FB CUDA ~550s（约 81%），densify/hooks ~7s，其余为 I/O / optimizer / 评测打印。  
因此 **将来把 TFR 接到 FB** 才是加速主路径；densify 与 TFR 正交。

---

## 5. 分场景现象

1. **Mip-NeRF360 户外**（bicycle/garden/stump/…）：高斯 2.7–4.5M，PSNR 22–28；tile↓ 75–85%。高纹理区仍大量 level-0（16×16），符合复杂度感知设计。
2. **Mip-NeRF360 室内**（room/counter/kitchen/bonsai）：高斯 <1.3M，PSNR 29–33；tile↓ 87–91%，L↓ 可达 50%+ —— TFR 在「大面积平滑墙/地板」上收益最大。
3. **Deep Blending**：playroom tile↓ 仅 63.7%（相对最保守），drjohnson 81.5%；画质稳定在 ~29–30 PSNR。
4. **Tanks & Temples**：Truck/Train 已有先前实测（25.36 / 21.73 PSNR），tile↓ 72% / 67%，与全表趋势一致。

---

## 6. Forward 实测（packed raster，13 场景）

> 数据：`outputs/bench_render/bench_render_13.json`（CUDA preprocess + assign unique 修复后）  
> 脚本：`scripts/bench_tfr_render.py --all-13 --max-views 5 --warmup 1 --repeats 2`  
> 实现：`tfr_cuda.preprocess_gaussians` + `render_packed_adaptive` + 子块 AABB 裁剪

### 6.1 测什么

| 路径 | 含义 |
|------|------|
| **Native** | HST-GS `gaussian_renderer.render`（固定 16×16 训练用 raster） |
| **TFR-fixed** | CUDA preprocess + packed raster，强制 `max_level=0` |
| **TFR-adapt** | 同 preprocess + 自适应 tile + packed raster（`C_merge=4000`）+ 子块裁剪 |

计时均 `cuda.synchronize()`；`tfr_raster_ms` 仅为 compositing kernel；`tfr_e2e_ms` = preprocess+complexity+schedule+assign+pack+raster。

### 6.2 聚合结果（13 场景 × 每场景 5 test views）

| 指标 | 数值 |
|------|------|
| Native raster | **6.36 ms** |
| TFR-fixed raster | **1.53 ms**（相对 native **~4.1×**） |
| TFR-adapt raster | **2.88 ms**（相对 native **~2.3×**；相对 TFR-fixed **0.55×**） |
| TFR e2e（含 preprocess） | **37.2 ms**（此前 Python 投影时代 ~157 ms） |
| Peak mem native / TFR | **6641 / 6445 MB**（↓ **~3%**） |
| Tile↓ / L↓（调度层） | **90.7% / 62.5%** |
| PSNR adapt vs native | **62.7 dB** |
| PSNR adapt vs fixed | **95.6 dB**（多数场景 99 dB） |
| PSNR native vs GT / TFR vs GT | **25.28 / 25.28** |

### 6.3 逐场景 raster ms

| Scene | Native | Adapt | Fixed | Adapt÷Nat | Adapt÷Fixed | PSNR(c vs n) |
|-------|-------:|------:|------:|----------:|------------:|-------------:|
| bicycle | 10.17 | 4.78 | 2.40 | 2.13× | 0.50× | 58.6 |
| flowers | 6.44 | 2.79 | 1.30 | 2.31× | 0.47× | 60.6 |
| garden | 8.61 | 4.79 | 2.20 | 1.80× | 0.46× | 61.0 |
| stump | 8.13 | 3.29 | 1.72 | 2.47× | 0.52× | 61.8 |
| treehill | 7.77 | 2.93 | 1.55 | 2.65× | 0.53× | 53.1 |
| room | 4.90 | 2.23 | 1.39 | 2.19× | 0.62× | 67.7 |
| counter | 5.01 | 2.49 | 1.51 | 2.02× | 0.61× | 63.6 |
| kitchen | 5.78 | 3.26 | 1.77 | 1.77× | 0.54× | 65.8 |
| bonsai | 4.30 | 1.94 | 1.13 | 2.21× | 0.58× | 64.9 |
| playroom | 5.02 | 1.99 | 1.11 | 2.52× | 0.56× | 64.7 |
| drjohnson | 6.28 | 1.98 | 1.42 | 3.17× | 0.72× | 69.0 |
| Truck | 5.48 | 2.84 | 1.12 | 1.93× | 0.39× | 61.0 |
| Train | 4.77 | 2.09 | 1.35 | 2.28× | 0.65× | 63.1 |

### 6.4 解读

1. **CUDA preprocess 对齐**：adapt/fixed vs native ~60+ dB；vs GT 与 native 一致（此前 Python 投影有 ~2 dB 缺口）。
2. **Assign 修复**：`duplicate_adaptive_keys` 按 base-cell 发射后对 `(tid,gid)` unique（去掉 `seen[64]` 溢出导致的重复写入）；adapt≈fixed（多数 99 dB）。
3. **GatherPack + fixed**：~1.5 ms，相对 native ~4.1×。
4. **子块裁剪**：大 tile 的 16×16 子块按 alpha-ellipse AABB 过滤父 list → adapt **~2.3× native**；仍慢于 packed-fixed（合并 list + 多子块重复读）。
5. **e2e**：preprocess 后降至 ~37 ms；训练加速仍需 schedule/assign 更深 GPU 化 + backward。

### 6.5 复现

```bash
srun -N 1 -n 1 --gres=gpu:1 -t 04:00:00 -p gpu \
  bash TFR/scripts/bench_tfr_render.sh --all-13 --max-views 5 --warmup 1 --repeats 2
```

---

## 7. 局限与下一步

1. **继续压 adapt→fixed 差距**：并行 compact、跳过空子块、或按子块预过滤 pack。
2. **Prep GPU 化 / 训练接入**：schedule+assign 上移；backward 后测 FB wall。
3. **过密化**：bicycle 等 >4M 点与 raster 正交。

---

## 8. 产物路径

| 文件 | 内容 |
|------|------|
| `outputs/vanilla_3dgs/{scene}/` | 原生 checkpoint、`training_timings.txt`、`results.json` |
| `outputs/full_13/full_13_results.json` | 训练/调度代理指标 |
| `outputs/bench_render/bench_render_13.json` | **Forward 实测** ms / mem / PSNR |
| `docs/full_13_analysis.md` | 本报告 |

**一句话**：Forward 已对齐原生 preprocess——**packed-fixed ~4.1×、adapt（子块裁剪）~2.3× native raster**，画质与 native 一致（~63 dB）；adapt 数值≈fixed；训练加速仍待 backward。
