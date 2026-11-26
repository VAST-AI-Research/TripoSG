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

### AETHRYX Studio desktop app (provisional identity)

The visible product name is **AETHRYX Studio** (provisional); **NIKIX** is the working brand. The tagline and product metadata are centralized in `studio_metadata.py`. TripoSG remains the generation engine. No legal entity, trademark registration, or product version is claimed.

Start the Windows app with `.\iniciar-aethryx-studio.bat` (the previous `iniciar-triposg.bat` remains as an alias). Select a reference image and output path, set inference steps, face limit, seed and guidance, then choose **Generar modelo 3D**. The app reports the actual output from the model process, supports cancellation, and opens a successful GLB in the viewer. Use mouse drag to rotate, the wheel to zoom, and the Matplotlib toolbar to pan or reset the camera. Import GLB, GLTF, OBJ, STL, or PLY files with **Abrir malla**.

Use **Imagen > Analizar hoja de personaje** to draw reference regions, label their view or detail type, and optionally designate one full-body view as the primary input. The analyzer can now run the locally installed OpenCV Haar frontal-face detector and add face/eye regions as unconfirmed proposals. Those proposals must be manually confirmed or reassigned; Haar does not classify full-body views or detect clothing, hair, mouth, or accessories. Regions and normalized crop coordinates are stored in `.aeth` projects. The current TripoSG engine receives only the approved primary crop and does not fuse multiple views. Without a primary crop, the app asks before sending the whole sheet as a single image.

See `AETHRYX_VISION_PHASE1.md` for the implemented scope, unavailable capabilities, engine/tool contracts, and commercial license review notes.

See `AETHRYX_VISION_PHASE2.md` for background analysis, manual reclasification and crop replacement, optional MediaPipe pose/face providers, typed KIRIA tool context, local diagnostics, and explicit remaining multiview/GPU limitations.

See `AETHRYX_VISION_PHASE3.md` for NumPy-safe bbox validation, deterministic primary-view selection, MediaPipe Tasks status, optional local ONNX segmentation, multiview engine evaluation, and the honest GPU viewer status.

See `AETHRYX_VISION_PHASE4.md` and `test_assets/anime/` for the real local anime
test, PyVista preview, TripoSG GLB metrics, license review, defects, and exact
VRAM measurements. Weights remain excluded from Git.

See `AETHRYX_ADAPTIVE_ENGINE.md` for hardware profiles, FP16/INT8/INT4 status,
CPU offloading, CUDA OOM recovery, basic edit mode and benchmark requirements.

Projects use a versioned `.aeth` JSON manifest containing generation settings and external image/model paths; source assets are not copied into the project. The manifest does not yet store mesh edits, materials, skeletons or animations. Keep referenced files available when moving a project.

The scene list supports object selection, visibility toggling, and isolating selected objects. The View menu provides front, side, rear, and top cameras, orthographic/perspective projection, grid, solid/wireframe display, and frame-all. Mesh statistics are computed from the source scene. Preview decimation works on copies and is capped at 8,000 faces per object for responsive rendering; node transforms and the original source scene are preserved. **Exportar escena original como GLB** writes the original GLB bytes unchanged, or exports the source scene for other formats. The startup GPU check uses `nvidia-smi`; PyTorch is loaded only when opening its detailed diagnostics or generating a model.

Menus expose the current features. Automatic reference recognition, multi-view reconstruction, perspective synthesis, undo/redo, vertex/edge/face editing, material/UV tools, mesh repair/retopology, rigging, and animation are explicitly marked as upcoming and are not implemented.

An optional Windows launcher can be built with `pip install pyinstaller` followed by `pyinstaller --noconsole --name AETHRYXStudio AETHRYXStudio.py`. It is not a standalone distribution: it still expects the repository, `.venv`, and downloaded model weights. The batch launcher is the simplest way to start the local app.

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

## Product scope and limitations

This is an experimental local front end and does not require Blender. Generated meshes are not rigged, do not include Unity/VRChat avatar setup, and have not been validated as upload-ready avatars. The viewer can approximate basic per-mesh colors but does not render complete PBR materials or textures. CPU-only generation, broad hardware coverage, and the optional Flash decoder are not validated; the Windows path uses marching cubes because the optional `diso` dependency is not available in the tested Windows setup. First use needs internet access to download model weights. Check upstream model and asset licenses before redistribution or commercial use.

## Validation performed

On Windows 10 with Python 3.10.11 and an NVIDIA RTX 5070 (CUDA 12.8):

- `python -m pip check` — passed with no broken requirements.
- `python -m unittest discover -s tests` — 9 project-manifest, GLB-validation, and reference-crop tests passed.
- `python -m py_compile app.py character_sheet.py character_sheet_dialog.py studio_project.py studio_metadata.py AETHRYXStudio.py` — passed.
- Generated and opened a GLB from the repository's sample image on the GPU; the face-limited test mesh loaded as 2,506 vertices and 5,000 faces.
- Loaded that GLB into the desktop 3D viewer and confirmed it rendered.
- Loaded the existing 1,384,278-face `output.glb` in the viewer; it displayed as an 8,000-face preview while retaining source statistics and scene data.
- Manually selected, labeled, and designated a primary crop from a synthetic character sheet; verified that the generation worker receives the crop instead of the full sheet.
- Saved and reopened `.aeth` project settings and reference metadata; confirmed that copying an imported GLB preserves its bytes.
- Tested a translated GLB scene: viewer transforms were respected, source geometry stayed unchanged, and selection, visibility, camera, wireframe, and solid controls worked.
- Confirmed the Python GUI worker's output can be captured without opening a console window; built and launched the no-console executable launcher.

There is not yet an automated test suite for the desktop UI, CPU inference, other GPU types, full material fidelity, or Unity/VRChat avatar compatibility.

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
