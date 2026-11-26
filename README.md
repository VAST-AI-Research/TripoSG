# TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models

<div align="center">

[![Project Page](https://img.shields.io/badge/🏠-Project%20Page-blue.svg)](https://yg256li.github.io/TripoSG-Page/)
[![Paper](https://img.shields.io/badge/📑-Paper-green.svg)](https://arxiv.org/abs/2502.06608)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-TripoSG-yellow.svg)](https://huggingface.co/VAST-AI/TripoSG)
[![Online Demo](https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Space-TripoSG-blue)](https://huggingface.co/spaces/VAST-AI/TripoSG)
[![Online Demo](https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Space-TripoSG%20scribble-blue)](https://huggingface.co/spaces/VAST-AI/TripoSG-scribble)

**By [Tripo](https://www.tripo3d.ai)**

</div>

![teaser](assets/doc/triposg_teaser.png)

TripoSG is an advanced high-fidelity, high-quality and high-generalizability image-to-3D generation foundation model. It leverages large-scale rectified flow transformers, hybrid supervised training, and a high-quality dataset to achieve state-of-the-art performance in 3D shape generation.

## ✨ Key Features

- **High-Fidelity Generation**: Produces meshes with sharp geometric features, fine surface details, and complex structures
- **Semantic Consistency**: Generated shapes accurately reflect input image semantics and appearance
- **Strong Generalization**: Handles diverse input styles including photorealistic images, cartoons, and sketches
- **Robust Performance**: Creates coherent shapes even for challenging inputs with complex topology

## 🔬 Technical Highlights

- **Large-Scale Rectified Flow Transformer**: Combines RF's linear trajectory modeling with transformer architecture for stable, efficient training
- **Advanced VAE Architecture**: Uses Signed Distance Functions (SDFs) with hybrid supervision combining SDF loss, surface normal guidance, and eikonal loss
- **High-Quality Dataset**: Trained on 2 million meticulously curated Image-SDF pairs, ensuring superior output quality
- **Efficient Scaling**: Implements architecture optimizations for high performance even at smaller model scales

## 🔥 Updates

* [2025-04] Release TripoSG-scribble, a CFG-distilled, 512 token model for fast shape prototyping from scribble+prompt! Try the online demo [here](https://huggingface.co/spaces/VAST-AI/TripoSG-scribble).
* [2025-03] Release of TripoSG 1.5B parameter rectified flow model and VAE trained on 2048 latent tokens, along with inference code and interactive demo

## 🔨 Installation

Clone the repo:
```bash
git clone https://github.com/VAST-AI-Research/TripoSG.git
cd TripoSG
```

Create a Python 3.10 environment (Conda or `venv`):
```bash
conda create -n tripoSG python=3.10
conda activate tripoSG
```

On Windows, a `venv` can be created from PowerShell:
```bash
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Install PyTorch for the CUDA version supported by your NVIDIA driver (CUDA 12.8 example), then install the project dependencies:
```bash
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Nikix 3D Studio desktop app (experimental)

Start the app with `.\iniciar-triposg.bat`. Select an image and output path, set the inference steps and face limit, then choose **Generate 3D model**. The app shows generation output and progress, and opens the resulting mesh in its interactive viewer. Use the mouse to rotate and zoom, the Matplotlib toolbar to pan/reset the view, or **Open 3D model...** to inspect an existing GLB, GLTF, OBJ, STL, or PLY file.

The generated `TripoSG.exe` launcher is optional and is not a standalone distribution: it expects the repository, `.venv`, and downloaded model weights to remain available. The batch launcher is the simplest way to start the local app.

The first generation downloads model weights from Hugging Face and therefore needs an internet connection. Weights are stored under `pretrained_weights/` and are intentionally excluded from Git.

## 💡 Quick Start

Generate a 3D mesh from an image:
```bash
python -m scripts.inference_triposg --image-input assets/example_data/hjswed.png --output-path ./output.glb
```

Limiting the number of faces:
```bash
python -m scripts.inference_triposg --image-input assets/example_data/hjswed.png --faces 5000 --output-path ./output.glb
```

or from scribble+prompt:
```bash
 python -m scripts.inference_triposg_scribble --image-input assets/example_scribble_data/cat_with_wings.png --prompt "a cat with wings" --scribble-conf 0.3 --output-path output.glb
```

The required model weights will be automatically downloaded:
- TripoSG (image condition) model from [VAST-AI/TripoSG](https://huggingface.co/VAST-AI/TripoSG) → `pretrained_weights/TripoSG`
- TripoSG-scribble (scribble+prompt condition) model from [VAST-AI/TripoSG-scribble](https://huggingface.co/VAST-AI/TripoSG-scribble) → `pretrained_weights/TripoSG-scribble`
- RMBG model from [briaai/RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4) → `pretrained_weights/RMBG-1.4`

## Implementation status

The Windows desktop app currently supports image-to-mesh generation, progress and output logs, GLB export, and interactive geometry preview/loading for common mesh formats. Mesh previews are simplified when needed to keep the viewer responsive.

This is an experimental local front end for TripoSG and does not require Blender, but it is not yet a complete avatar authoring tool. Generated meshes are not rigged, do not include Unity/VRChat avatar setup, and have not been validated as upload-ready VRChat avatars. The viewer previews geometry rather than faithfully rendering all source materials or textures. CPU-only generation, broad hardware coverage, and the optional Flash decoder are not validated; the Windows path uses the marching-cubes decoder because the optional `diso` dependency is not available in the tested Windows setup. Check the upstream model and asset licenses before redistribution or commercial use.

## Validation performed

On Windows 10 with Python 3.10.11 and an NVIDIA RTX 5070 (CUDA 12.8):

- `python -m pip check` — passed with no broken requirements.
- `python -m py_compile app.py TripoSG.py` — passed.
- Generated and opened a GLB from the repository's sample image on the GPU; the face-limited test mesh loaded as 2,506 vertices and 5,000 faces.
- Loaded that GLB into the desktop 3D viewer and confirmed it rendered.
- Built and launched the no-console Windows executable launcher.

There is not yet an automated test suite for the desktop UI, CPU inference, other GPU types, or Unity/VRChat avatar compatibility.

## 💻 System Requirements

- CUDA-enabled GPU with at least 8GB VRAM

## 📝 Tips

- If you want to use the full VAE module (including the encoder part), you need to uncomment the Line-15 in `triposg/models/autoencoders/autoencoder_kl_triposg.py` and install `torch-cluster`. and run:
```
python -m scripts.inference_vae --surface-input assets/example_data_point/surface_point_demo.npy
```

## 🤝 Community & Support

- **Issues & Discussions**: Use GitHub Issues for bug reports and feature requests.
- **Contributing**: We welcome contributions!

## 📚 Citation

```
@article{li2025triposg,
  title={TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models},
  author={Li, Yangguang and Zou, Zi-Xin and Liu, Zexiang and Wang, Dehu and Liang, Yuan and Yu, Zhipeng and Liu, Xingchao and Guo, Yuan-Chen and Liang, Ding and Ouyang, Wanli and others},
  journal={arXiv preprint arXiv:2502.06608},
  year={2025}
}
```

## ⭐ Acknowledgements

We would like to thank the following open-source projects and research works that made TripoSG possible:

- [DINOv2](https://github.com/facebookresearch/dinov2) for their powerful visual features
- [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4) for background removal
- [🤗 Diffusers](https://github.com/huggingface/diffusers) for their excellent diffusion model framework
- [HunyuanDiT](https://github.com/Tencent/HunyuanDiT) for DiT
- [FlashVDM](https://github.com/Tencent/FlashVDM) for their lightning vecset decoder
- [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet) for 3D shape representation

We are grateful to the broader research community for their open exploration and contributions to the field of 3D generation.
