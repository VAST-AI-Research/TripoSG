import argparse
import os
import sys
from glob import glob
from typing import Any, Union

import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from triposg.pipelines.pipeline_triposg_scribble import TripoSGScribblePipeline



@torch.no_grad()
def run_triposg_scribble(
    pipe: Any,
    image_input: Union[str, Image.Image],
    prompt: str,
    seed: int,
    num_inference_steps: int = 16,
    scribble_confidence: float = 0.4,
    prompt_confidence: float = 1.0
) -> trimesh.Scene:

    img_pil = Image.open(image_input).convert("RGB")

    outputs = pipe(
        image=img_pil,
        prompt=prompt,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=0, # this is a CFG-distilled model
        attention_kwargs={"cross_attention_scale": prompt_confidence, "cross_attention_2_scale": scribble_confidence},
        use_flash_decoder=False, # there're some boundary problems when using flash decoder with this model
        dense_octree_depth=8, hierarchical_octree_depth=8 # 256 resolution for faster inference
    ).samples[0]
    mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))    
    return mesh


if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float16

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-input", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="./output.glb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=16)
    # feel free to tune the scribble confidence, 0.3-0.5 often gives good results for hand-drawn sketches
    parser.add_argument("--scribble-conf", type=float, default=0.4)
    parser.add_argument("--prompt-conf", type=float, default=1.0)
    args = parser.parse_args()

    # download pretrained weights
    triposg_scribble_weights_dir = "pretrained_weights/TripoSG-scribble"
    snapshot_download(repo_id="VAST-AI/TripoSG-scribble", local_dir=triposg_scribble_weights_dir)

    # init tripoSG pipeline
    pipe: TripoSGScribblePipeline = TripoSGScribblePipeline.from_pretrained(triposg_scribble_weights_dir).to(device, dtype)

    # run inference
    run_triposg_scribble(
        pipe,
        image_input=args.image_input,
        prompt=args.prompt,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        scribble_confidence=args.scribble_conf,
        prompt_confidence=args.prompt_conf,
    ).export(args.output_path)
    print("Mesh saved to", args.output_path)
