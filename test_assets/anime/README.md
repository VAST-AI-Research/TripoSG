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

These artifacts are visual QA evidence. They are not a ready VRChat avatar:
the mesh is not rigged, skinned or validated for Unity/VRChat, and TripoSG
produced visible geometry defects in this test.
