# Anime visual test assets

The source image `authorized-anime-face.png` comes from
[jerryli27/AniSeg](https://github.com/jerryli27/AniSeg), whose repository
license is Apache-2.0. It is an authorized project sample, not a personal image.

The repository also carries Apache-2.0 code licenses for:

- MediaPipe Tasks
- SkyTNT anime-segmentation / ISNet
- AniSeg

## Generated evidence

- `authorized-anime-vision-review.png`: overlay of real MediaPipe/ONNX detections.
- `authorized-anime-alpha.png`: 8-bit character alpha mask produced by the
  SkyTNT/ISNet ONNX model.
- `authorized-anime-triposg.glb`: TripoSG 20-step test at 5,000 faces.
- `authorized-anime-triposg-preview.png`: PyVista/VTK render of the GLB.
- `genesis-75-original.glb`: 75-step original geometry, 342,252 vertices,
  684,448 faces, watertight.
- `genesis-75-front.png`, `genesis-75-side.png`, `genesis-75-rear.png`:
  real PyVista renders of the original 75-step mesh.
- `genesis-50-5000.glb`: 50-step no-decimation generation.
- `genesis-75-25000.glb`: 25,000-face copy of the original.
- `genesis-75-100000.glb`: 100,000-face copy of the original.

These artifacts are visual QA evidence. They are not a ready VRChat avatar:
the mesh is not rigged, skinned or validated for Unity/VRChat, and TripoSG
produced visible geometry defects in this test.
