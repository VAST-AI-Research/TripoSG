import argparse
import os
import sys
import tempfile
from typing import Any, Union

import gradio as gr
import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image

import matplotlib.cm as cm

# Add the scripts directory to sys.path to allow importing local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Now import from other scripts in the same directory
try:
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline
    from image_process import prepare_image
    from briarmbg import BriaRMBG
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure that 'triposg', 'image_process', and 'briarmbg' are correctly placed or installed.")
    sys.exit(1)

# --- Model Loading ---
device = "cuda"
dtype = torch.float16

print("Downloading and loading models...")
# Define directories for weights relative to the script location
weights_base_dir = os.path.join(script_dir, "..", "pretrained_weights") # Go up one level from scripts
triposg_weights_dir = os.path.join(weights_base_dir, "TripoSG")
rmbg_weights_dir = os.path.join(weights_base_dir, "RMBG-1.4")

os.makedirs(triposg_weights_dir, exist_ok=True)
os.makedirs(rmbg_weights_dir, exist_ok=True)

# Download pretrained weights if not present
if not os.path.exists(os.path.join(triposg_weights_dir, "config.json")):
     print(f"Downloading TripoSG weights to {triposg_weights_dir}...")
     snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_weights_dir, local_dir_use_symlinks=False)
else:
    print("TripoSG weights found.")

if not os.path.exists(os.path.join(rmbg_weights_dir, "config.json")):
    print(f"Downloading RMBG-1.4 weights to {rmbg_weights_dir}...")
    snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir, local_dir_use_symlinks=False)
else:
    print("RMBG-1.4 weights found.")

# Init rmbg model for background removal
print("Initializing BriaRMBG model...")
rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
rmbg_net.eval()
print("BriaRMBG model loaded.")

# Init tripoSG pipeline
print("Initializing TripoSGPipeline...")
pipe: TripoSGPipeline = TripoSGPipeline.from_pretrained(triposg_weights_dir).to(device, dtype)

print("TripoSGPipeline loaded.")
print("Models loaded successfully.")

# --- Background Removal Function ---
@torch.no_grad()
def remove_background(input_image: Image.Image, progress=gr.Progress(track_tqdm=True)) -> tuple[Image.Image, Image.Image]: # Modified return type
    """Removes the background from the input image and returns it twice (for display and state)."""
    if input_image is None:
        raise gr.Error("No input image provided. Please upload an image.")

    print("Preparing image (removing background)...")
    progress(0.1, desc="Removing background")
    # Save PIL image to a temporary file because prepare_image expects a path
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img_file:
        input_image.save(temp_img_file.name)
        # Call prepare_image specifically for background removal
        img_pil_processed = prepare_image(temp_img_file.name, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
    os.unlink(temp_img_file.name) # Clean up temporary file

    if isinstance(img_pil_processed, str): # Error handling from prepare_image
        raise gr.Error(f"Background removal failed: {img_pil_processed}")

    progress(1.0, desc="Background removed")
    print("Background removal complete.")
    # Return the image twice: once for the Image display, once for the State
    return img_pil_processed, img_pil_processed # Modified return

# --- Inference Function ---
@torch.no_grad()
def run_inference(
    input_image: Image.Image, # Added: Original input image
    processed_image: Image.Image, # From state (might be None)
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    seed: int = 42, # Renamed parameter to avoid confusion
    progress=gr.Progress(track_tqdm=True)
) -> tuple[str, Any]: # Only returns GLB path and mesh object
    """
    Runs the TripoSG inference pipeline using either the pre-processed image
    or the original input image if no background removal was performed.
    """
    image_to_process = None
    if processed_image is not None:
        image_to_process = processed_image
        print("Using pre-processed (background removed) image for inference.")
    elif input_image is not None:
        image_to_process = input_image
        # Note: The pipeline expects an image with alpha channel or a white background
        # for best results. Using raw input might work but could be suboptimal.
        print("Using original input image for inference (background removal skipped).")
    else:
        # This case should ideally not happen if the button requires an input
        raise gr.Error("No input image provided. Please upload an image.")

    if seed == -1:
        seed = np.random.randint(0, 999999 + 1) # Generate a random seed
        print(f"Using randomly generated seed: {seed}")

    # Use the selected image
    img_pil_processed = image_to_process # Use the determined image

    print("Running TripoSG inference...")
    progress(0.1, desc="Starting Diffusion") # Adjusted progress start
    generator = torch.Generator(device=device).manual_seed(seed)

    outputs = pipe(
        image=img_pil_processed,
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        # No progress callback needed here anymore for diffusion steps within pipe
    ).samples[0]

    progress(0.9, desc="Creating Mesh") # Adjusted progress start
    vertices = outputs[0].astype(np.float64)
    faces = np.ascontiguousarray(outputs[1])

    if not np.issubdtype(faces.dtype, np.integer):
        print(f"Warning: Faces are not integers (dtype: {faces.dtype}). Attempting conversion.")
        faces = faces.astype(np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
         raise gr.Error(f"Invalid vertices shape: {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
         raise gr.Error(f"Invalid faces shape: {faces.shape}")
    if faces.max() >= len(vertices):
        raise gr.Error(f"Face index {faces.max()} out of bounds for {len(vertices)} vertices.")

    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        z_coords = mesh.vertices[:, 2]
        z_min, z_max = z_coords.min(), z_coords.max()
        if z_max - z_min > 1e-6:
            z_normalized = (z_coords - z_min) / (z_max - z_min)
        else:
            z_normalized = np.zeros_like(z_coords)

        cmap = cm.get_cmap('viridis')
        colors = (cmap(z_normalized) * 255).astype(np.uint8)
        mesh.visual.vertex_colors = colors
        print("Mesh created and colored successfully.")
    except Exception as e:
        raise gr.Error(f"Failed to create or color Trimesh object: {e}")

    progress(0.95, desc="Saving temporary GLB")
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as temp_glb_file:
        mesh.export(temp_glb_file.name, file_type='glb')
        output_glb_path = temp_glb_file.name
    print(f"Mesh saved temporarily to {output_glb_path}")

    progress(1.0, desc="Done")
    return output_glb_path, mesh

# --- Saving Functions ---
def save_mesh(mesh_obj: trimesh.Trimesh, file_format: str) -> str: # Return type is str (file path)
    """Saves the mesh object to a temporary file and returns the path."""
    if mesh_obj is None:
        raise gr.Error("No mesh generated yet. Please run inference first.")

    # Map format to suffix and trimesh file_type
    format_map = {
        "glb": (".glb", "glb"),
        "obj": (".obj", "obj"),
        "stl": (".stl", "stl"),
        "ply": (".ply", "ply")
    }
    if file_format not in format_map:
        raise gr.Error(f"Unsupported format: {file_format}")

    suffix, trimesh_type = format_map[file_format]

    # Create a temporary file - Gradio should manage its lifecycle when used with DownloadButton
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            mesh_obj.export(temp_file.name, file_type=trimesh_type)
            temp_path = temp_file.name
            print(f"Mesh exported temporarily as {file_format} to {temp_path}")
            # Return the path to the temporary file
            return temp_path
    except Exception as e:
        raise gr.Error(f"Failed to export mesh as {file_format} to temporary file: {e}")

# --- Gradio Interface ---
with gr.Blocks(title="TripoSG 3D Generation") as demo:
    gr.Markdown("# TripoSG: Image-to-3D Generation")
    gr.Markdown("Upload an image to generate a 3D model using the TripoSG pipeline.")

    # Store the generated mesh object for saving later
    mesh_state = gr.State(None)
    # Store the processed image for display
    processed_image_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Input Image")
            remove_bg_btn = gr.Button("Remove Background", variant="secondary") # New Button
            processed_image_display = gr.Image(type="pil", label="Processed Image (Background Removed)", interactive=False)
            num_inference_steps = gr.Slider(minimum=10, maximum=200, value=50, step=1, label="Number of Inference Steps")
            guidance_scale = gr.Slider(minimum=1.0, maximum=15.0, value=7.0, step=0.1, label="Guidance Scale")
            seed = gr.Slider(minimum=-1, maximum=999999, value=-1, step=1, label="Seed") # Removed randomize=True
            submit_btn = gr.Button("Generate 3D Model", variant="primary")

        with gr.Column(scale=2): # Increased scale for output column
            output_model = gr.Model3D(label="Generated 3D Model", camera_position=(30, 60, 1.5), height = 1000 )
            gr.Markdown("Download:")
            with gr.Row():
                file_format_dropdown = gr.Dropdown(
                    choices=["glb", "obj", "stl", "ply"],
                    value="stl",
                    label="Select Format",
                    interactive=True
                )
                download_btn = gr.DownloadButton("Download Model", variant="secondary")

    # Connect Remove Background button
    remove_bg_btn.click(
        fn=remove_background,
        inputs=[input_image],
        outputs=[processed_image_display, processed_image_state] # Update display and state
    )

    # Connect Generate 3D Model button (takes processed image state as input)
    submit_btn.click(
        fn=run_inference, # Use the modified inference function
        inputs=[input_image, processed_image_state, num_inference_steps, guidance_scale, seed],  # Input is processed image state
        outputs=[output_model, mesh_state] # Output GLB path to Model3D, mesh object to state
    )

    # Connect Download button
    download_btn.click(
        fn=save_mesh,
        inputs=[mesh_state, file_format_dropdown],
        outputs=[download_btn]
    )


if __name__ == "__main__":
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Run TripoSG Gradio App")
    parser.add_argument("--share", action="store_true", help="Create a public link (use with caution)")
    parser.add_argument(
        "--port",
        type=int,
        default=None, # Let Gradio choose default port if not specified
        help="Port number to run the Gradio server on"
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default=None, # Default to 127.0.0.1 (localhost) unless specified
        help="Server name/IP address to bind to (e.g., '0.0.0.0' for all interfaces)"
    )
    args = parser.parse_args()

    # Launch the Gradio app with parsed arguments
    print(f"Launching Gradio App...")
    print(f"  Share: {args.share}")
    if args.server_name:
        print(f"  Server Name: {args.server_name}")
    if args.port:
        print(f"  Port: {args.port}")

    demo.launch(
        share=args.share,
        server_port=args.port, # Pass the port argument here
        server_name=args.server_name # Pass the server_name argument here
    )