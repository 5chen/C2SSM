## [CVPR 2026] Scan Clusters, Not Pixels: A Cluster-Centric Paradigm for Efficient Ultra-high-definition Image Restoration

[Chen Wu](https://github.com/5chen)<sup>1</sup>, [Ling Wang](https://daviswang0.github.io/)<sup>2</sup>, [Zhuoran Zheng](https://scholar.google.com.hk/citations?user=pXzPL-sAAAAJ&hl=zh-CN)<sup>3</sup>, [Yuning Cui](https://www.ce.cit.tum.de/en/air/people/yuning-cui/)<sup>4</sup>, [Zhixiong Yang]()<sup>1</sup>, [Xiangyu Chen](https://chxy95.github.io/)<sup>5</sup>, [Yue Zhang]()<sup>6</sup>, [Weidong Jiang](https://xplorestaging.ieee.org/author/37288834600)<sup>1</sup>, [Jingyuan Xia](https://www.xiajingyuan.com/)<sup>1,†</sup>

<sup>1</sup>National University of Defense Technology, <sup>2</sup>HKUST(GZ), <sup>3</sup>Qilu University of Technology, <sup>4</sup>Technical University of Munich, <sup>5</sup>Institute of Artificial Intelligence (TeleAI), <sup>6</sup>Beihang University

† Corresponding author

**CVPR 2026** | [arXiv](https://arxiv.org/abs/2602.21917) | [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

<p align="center">
  <img src="figs/framework.png" width="90%">
</p>

Ultra-High-Definition (UHD) image restoration is trapped in a scalability crisis: existing models, bound to pixel-wise operations, demand unsustainable computation. While state space models (SSMs) like Mamba promise linear complexity, their pixel-serial scanning remains a fundamental bottleneck for the millions of pixels in UHD content.

**C2SSM** introduces a cluster-centric visual state space model that shifts from pixel-serial to cluster-serial scanning. The rich feature distribution of a UHD image is distilled into a sparse set of semantic centroids via a neural-parameterized mixture model. C2SSM scans and reasons over a handful of cluster centers, then diffuses the global context back to all pixels through a principled similarity distribution, while a lightweight modulator preserves fine details.

<p align="center">
  <img src="figs/diff.png" width="85%">
</p>

### Key Components
- **Cluster-Centric Scanning Module (CCSM)**: Models feature distribution via learnable cluster centroids, applies SSM only to sparse centroids, and propagates context via similarity-guided score diffusion.
- **Spatial-Channel Feature Modulator (SCFM)**: Lightweight parallel path that preserves high-frequency details attenuated during clustering.

## Results

### UHD-Blur

| Methods | Type | Venue | PSNR | SSIM | Param |
|---------|------|-------|------|------|-------|
| MIMO-Unet++ | non-UHD | ICCV'21 | 25.03 | 0.752 | 16.1M |
| Restormer | non-UHD | CVPR'22 | 25.21 | 0.752 | 26.1M |
| FFTformer | non-UHD | CVPR'23 | 25.41 | 0.757 | 16.6M |
| UHDformer | UHD | AAAI'24 | 28.82 | 0.844 | 0.34M |
| DreamUHD | UHD | AAAI'25 | 29.33 | 0.852 | 1.45M |
| ERR | UHD | CVPR'25 | 29.72 | 0.861 | 1.13M |
| **C2SSM (Ours)** | **UHD** | **CVPR'26** | **31.53** | **0.890** | **2.71M** |

### 4K-Rain13k

| Methods | Type | Venue | PSNR | SSIM | Param |
|---------|------|-------|------|------|-------|
| Restormer | non-UHD | CVPR'22 | 33.02 | 0.934 | 26.12M |
| DRSformer | non-UHD | CVPR'23 | 32.96 | 0.933 | 33.65M |
| NeRD-Rain | non-UHD | CVPR'24 | 33.63 | 0.935 | 22.9M |
| MambaIRv2 | non-UHD | CVPR'25 | 33.17 | 0.939 | 12.7M |
| UDR-Mixer | UHD | TMM'25 | 34.30 | 0.951 | 4.90M |
| ERR | UHD | CVPR'25 | 34.48 | 0.952 | 1.13M |
| **C2SSM (Ours)** | **UHD** | **CVPR'26** | **35.13** | **0.956** | **2.71M** |

### UHD-Haze

| Methods | Type | Venue | PSNR | SSIM | Param |
|---------|------|-------|------|------|-------|
| Restormer | non-UHD | CVPR'22 | 12.72 | 0.693 | 26.11M |
| MB-TaylorFormer | non-UHD | ICCV'23 | 20.99 | 0.919 | 2.7M |
| UHDformer | UHD | AAAI'24 | 22.59 | 0.942 | 0.34M |
| UHD-processer | UHD | CVPR'25 | 23.24 | **0.953** | 1.6M |
| **C2SSM (Ours)** | **UHD** | **CVPR'26** | **24.08** | **0.942** | **2.71M** |

### UHD-LOL4K

| Methods | Type | Venue | PSNR | SSIM | Param |
|---------|------|-------|------|------|-------|
| Restormer | non-UHD | CVPR'22 | 36.90 | 0.988 | 26.11M |
| UHDFour | UHD | ICLR'23 | 36.12 | 0.990 | 17.54M |
| LLFormer | UHD | AAAI'23 | 37.33 | 0.988 | 24.52M |
| Wave-Mamba | UHD | MM'24 | 37.43 | 0.990 | 1.25M |
| MixNet | UHD | NeuroC'24 | 39.22 | **0.992** | 7.77M |
| D2Net | UHD | WACV'25 | 37.73 | **0.992** | 5.22M |
| **C2SSM (Ours)** | **UHD** | **CVPR'26** | **39.61** | **0.992** | **2.71M** |

### UHD-Snow

| Methods | Type | Venue | PSNR | SSIM | Param |
|---------|------|-------|------|------|-------|
| Restormer | non-UHD | CVPR'22 | 24.14 | 0.869 | 26.12M |
| UHDformer | UHD | AAAI'24 | 36.61 | 0.988 | 0.34M |
| UHDDIP | UHD | TCSVT'25 | 41.56 | **0.990** | 0.81M |
| **C2SSM (Ours)** | **UHD** | **CVPR'26** | **42.45** | **0.990** | **2.71M** |

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.12
- CUDA >= 11.3

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Mamba

```bash
pip install causal_conv1d
pip install mamba_ssm
```

## Dataset Preparation

Download the datasets and organize them as follows:

```
datasets/
├── uhddeblur/
│   ├── train/
│   │   ├── gt/
│   │   └── input/
│   └── test/
│       ├── gt/
│       └── input/
├── 4K-Rain13k/
│   ├── train/
│   │   ├── target/
│   │   └── input/
│   └── test/
│       ├── target/
│       └── input/
├── uhdhaze/
│   ├── train/
│   │   ├── gt/
│   │   └── input/
│   └── test/
│       ├── gt/
│       └── input/
└── lol4k/
    ├── train/
    │   ├── high/
    │   └── low/
    └── test/
        ├── high/
        └── low/
```

## Training

```bash
# Deblurring
bash train.sh uhdblur 0,1

# Deraining
bash train.sh uhdrain 0,1

# Dehazing
bash train.sh uhdhaze 0,1

# Low-light Enhancement
bash train.sh uhdlol 0,1
```

Or run directly:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=4337 \
    basicsr/train.py -opt options/train_C2SSM_uhdblur.yml --launcher pytorch
```

## Inference

```bash
python inference.py \
    -i <input_folder> \
    -g <gt_folder> \
    -w <model_weight_path> \
    -o <output_folder> \
    --device cuda:0 \
    --save_img
```

### Example

```bash
# Deblurring
python inference.py \
    -i datasets/uhddeblur/test/input \
    -g datasets/uhddeblur/test/gt \
    -w pretrained_models/C2SSM_uhdblur.pth \
    --save_img

# Deraining
python inference.py \
    -i datasets/4K-Rain13k/test/input \
    -g datasets/4K-Rain13k/test/target \
    -w pretrained_models/C2SSM_uhdrain.pth \
    --save_img
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{wu2026c2ssm,
    title={Scan Clusters, Not Pixels: A Cluster-Centric Paradigm for Efficient Ultra-high-definition Image Restoration},
    author={Wu, Chen and Wang, Ling and Zheng, Zhuoran and Cui, Yuning and Yang, Zhixiong and Chen, Xiangyu and Zhang, Yue and Jiang, Weidong and Xia, Jingyuan},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2026}
}
```

## Acknowledgments

This codebase is built upon [BasicSR](https://github.com/XPixelGroup/BasicSR), [Wave-Mamba](https://github.com/AlexZou14/Wave-Mamba), and [Mamba](https://github.com/state-spaces/mamba). We thank the authors for their excellent work.
