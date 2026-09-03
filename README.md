

# **HST-GS: Fast 3D Gaussian Splatting with Hierarchical Super-Tiles and Scheduled Refinement**
### [arXiv](#) | [Project Page](#)



HST-GS is an efficiency-oriented 3D Gaussian Splatting framework. It accelerates both
training and rendering while preserving the image formation model of the original 3DGS,
through three key components:

- **Hierarchical Super-Tile (HST) screening** — prunes splat-tile pairs with 4×4 super-tile test to cut rasterization cost.
- **Compact depth keys** — reduces the radix-sort key width (e.g. 24-bit) to speed up and
shrink sorting with negligible quality loss.
- **Scheduled Refinement (SR)** — skips redundant full forward-backward steps with a
piecewise stride schedule for a higher effective training speedup.

An optional eval-time **TFR** fixed-tile splatting backend further boosts rendering FPS.

## Installation

Our environment is the same as the original 3DGS. We tested it with Python 3.8, PyTorch and CUDA 11.8.

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate gaussian_splatting
```

The `environment.yml` already installs three of the CUDA submodules via pip editable installs
(`submodules/diff-gaussian-rasterization`, `submodules/simple-knn`, `submodules/fused-ssim`).
If you prefer to install the submodules manually (or the environment already exists), run:

### 2. Build the CUDA submodules

Training requires these three submodules:

```bash
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install ./submodules/fused-ssim
```

The optional eval-time TFR accelerator (used only by `render.py --use_tfr`) is built separately
on a machine with a GPU:

```bash
cd submodules/tfr/cuda
python setup.py build_ext --inplace
cd ../../..
```

See `[submodules/TFR.md](submodules/TFR.md)` for details.

## Dataset



### Mip-NeRF 360 Dataset

Please download the Mip-NeRF 360 dataset processed by colmap from [Mip-NeRF 360](https://jonbarron.info/mipnerf360/), and after unzipping "Dataset Pt. 1" and "Dataset Pt. 2", combine the scenes. Finally, the current directory should contain the following folders:

```
HST-GS
|---data
    |---Mip-NeRF360
        |---bicycle
        |   |---images
        |   |   |---<image 0>
        |   |   |---<image 1>
        |   |   |---...
        |   |---images_2
        |   |---images_4
        |   |---images_8
        |   |---sparse
        |       |---0
        |           |---cameras.bin
        |           |---images.bin
        |           |---points3D.bin
        |---bonsai
        |---...
```



### Tanks and Temples Dataset

Please download the "image set" of the 'Train' and 'Truck' scenes from the Tanks and Temples dataset from [Tanks and Temples](https://www.tanksandtemples.org/download/).

```
HST-GS
|---data
    |---Tanks_Temples
        |---Train
        |   |---images
        |   |   |---<image 0>
        |   |   |---...
        |   |---sparse
        |       |---0
        |           |---cameras.bin
        |           |---images.bin
        |           |---points3D.bin
        |---Truck
        |---...
```



### Deep Blending Dataset

```
HST-GS
|---data
    |---Deep_Blending
        |---drjohnson
        |   |---images
        |   |   |---<image 0>
        |   |   |---...
        |   |---sparse
        |       |---0
        |           |---cameras.bin
        |           |---images.bin
        |           |---points3D.bin
        |---playroom
        |---...
```



### Your Own Dataset

Our method requires the same data format as 3DGS. For your own data, you can use the processing method found in the ["Processing your own Scenes"](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes) section of the original 3DGS code.

## Training and Evaluation

A convenience script is provided to train and evaluate all scenes:

```bash
bash ./run.sh
```

For training and testing individual scenes, the commands are identical to the original 3DGS code:

```bash
# Training (with evaluation on the test split)
python train.py -s <path to scene> -m <path to output model> --eval

# Rendering the test/train splits of a trained model
python render.py -m <path to output model>

# Computing metrics (PSNR / SSIM / LPIPS) for rendered images
python metrics.py -m <path to output model>
```

For details, please refer to [Running](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#running) and [Evaluation](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#evaluation) of the original 3DGS repository.

## Viewer

Since the rendering process and point cloud storage format of our method are identical to those of the original 3DGS, our method can use the same viewer as the original 3DGS. For specific usage tutorials, please refer to [Interactive Viewers](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#interactive-viewers).

## License

This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and is released under the
Gaussian-Splatting License (Inria / MPII non-commercial research license). See [LICENSE.md](LICENSE.md) for the full text.
Commercial use requires prior consent from Inria.

## Acknowledgements

This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). Please follow the license of 3DGS. We thank all the authors for their great work and repos.

## Citation



If you find this repo helpful, please cite our paper.

```
@article{hstgs2026,
  title={HST-GS: Hierarchical Super-Tile Gaussian Splatting},
  author={Zhou, Zheng and Xiong, Yu-Jie and Zhang, Jia-Chen and Xia, Chun-Ming and Qiu, Xihe and Zhan, Hongjian},
  year={2026}
}
```

