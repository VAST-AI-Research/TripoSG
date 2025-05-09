import argparse
import numpy as np
import torch
import trimesh

from triposg.inference_utils import hierarchical_extract_geometry
from triposg.models.autoencoders import TripoSGVAEModel
from huggingface_hub import snapshot_download



def load_surface(data_path, num_pc=204800):
    data = np.load(data_path, allow_pickle=True).tolist()
    surface = data["surface_points"]  # Nx3
    normal = data["surface_normals"]  # Nx3

    rng = np.random.default_rng()
    ind = rng.choice(surface.shape[0], num_pc, replace=False)
    surface = torch.FloatTensor(surface[ind])
    normal = torch.FloatTensor(normal[ind])
    surface = torch.cat([surface, normal], dim=-1).unsqueeze(0)

    return surface


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    dtype = torch.float16 if device != "cpu" else torch.float32

    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-input", type=str, required=True)
    args = parser.parse_args()

    # download pretrained weights
    triposg_weights_dir = "pretrained_weights/TripoSG"
    snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_weights_dir)

    vae: TripoSGVAEModel = TripoSGVAEModel.from_pretrained(
        triposg_weights_dir,
        subfolder="vae",
    ).to(device, dtype=dtype)

    # load surface from sdf and encode
    surface = load_surface(
        args.surface_input, num_pc=204800
    ).to(device, dtype=dtype)
    sample = vae.encode(surface).latent_dist.sample()
    
    # vae infer 
    with torch.no_grad():
        geometric_func = lambda x: vae.decode(sample, sampled_points=x).sample
        output = hierarchical_extract_geometry(
            geometric_func,
            device,
            bounds=(-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
            dense_octree_depth=8,
            hierarchical_octree_depth=9,
        )
        meshes = [trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1]) for mesh_v_f in output]

    meshes[0].export("test_vae.glb")

