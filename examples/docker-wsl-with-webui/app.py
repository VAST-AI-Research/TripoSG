import os
import gradio as gr
import torch
from scripts.inference_triposg import run_triposg
from scripts.inference_triposg_scribble import run_triposg_scribble
from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from triposg.pipelines.pipeline_triposg_scribble import TripoSGScribblePipeline
from briarmbg import BriaRMBG
from huggingface_hub import snapshot_download
import uuid


device = "cuda"
dtype = torch.float16

rmbg_net = None
pipe = None
pipe_scribble = None


def download_models():
    triposg_weights_dir = "pretrained_weights/TripoSG"
    triposg_scribble_weights_dir = "pretrained_weights/TripoSG-scribble"
    rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
    
    if not os.path.exists(triposg_weights_dir):
        snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_weights_dir)
    if not os.path.exists(triposg_scribble_weights_dir):
        snapshot_download(repo_id="VAST-AI/TripoSG-scribble", local_dir=triposg_scribble_weights_dir)
    if not os.path.exists(rmbg_weights_dir):
        snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)


def init_models():
    global rmbg_net, pipe, pipe_scribble
    
    if rmbg_net is None:
        rmbg_net = BriaRMBG.from_pretrained("pretrained_weights/RMBG-1.4").to(device)
        rmbg_net.eval()

    if pipe is None:
        pipe = TripoSGPipeline.from_pretrained("pretrained_weights/TripoSG").to(device, dtype)
    
    if pipe_scribble is None:
        pipe_scribble = TripoSGScribblePipeline.from_pretrained("pretrained_weights/TripoSG-scribble").to(device, dtype)
    
    return rmbg_net, pipe, pipe_scribble


def generate_3d_from_image(image, seed, num_inference_steps, guidance_scale, faces, session_id=None):
    global rmbg_net, pipe
    if rmbg_net is None or pipe is None:
        rmbg_net, pipe, _ = init_models()

    sid = str(uuid.uuid4()) if session_id is None else session_id
    image_path = os.path.join(os.getcwd(), f"input_{sid}.png")
    image.save(image_path)
    
    mesh = run_triposg(
        pipe,
        image_input=image_path,
        rmbg_net=rmbg_net,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        faces=faces,
    )
    
    output_path = os.path.join(os.getcwd(), f"output_{sid}.glb")
    mesh.export(output_path)
    return output_path


def generate_3d_from_scribble(image, prompt, seed, num_inference_steps, scribble_conf, prompt_conf, session_id=None):
    global pipe_scribble
    if pipe_scribble is None:
        _, _, pipe_scribble = init_models()

    sid = str(uuid.uuid4()) if session_id is None else session_id
    image_path = os.path.join(os.getcwd(), f"input_{sid}.png")
    image.save(image_path)
    
    mesh = run_triposg_scribble(
        pipe_scribble,
        image_input=image_path,
        prompt=prompt,
        seed=seed,
        num_inference_steps=num_inference_steps,
        scribble_confidence=scribble_conf,
        prompt_confidence=prompt_conf,
    )
    
    output_path = os.path.join(os.getcwd(), "output_scribble.glb")
    mesh.export(output_path)
    return output_path


def create_interface():
    with gr.Blocks(title="TripoSG WebUI") as demo:
        gr.Markdown("# TripoSG WebUI")
        
        with gr.Tabs():
            with gr.TabItem("Image to 3D"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="pil", label="Input Image")
                        seed = gr.Slider(minimum=0, maximum=1000, value=42, step=1, label="Seed")
                        num_inference_steps = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Inference Steps")
                        guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, value=7.0, step=0.1, label="Guidance Scale")
                        faces = gr.Slider(minimum=-1, maximum=10000, value=-1, step=100, label="Number of Faces (-1 for no reduction)")
                        generate_btn = gr.Button("Generate 3D Model")
                    
                    with gr.Column():
                        output_model = gr.Model3D(label="Generated 3D Model")
                
                generate_btn.click(
                    fn=generate_3d_from_image,
                    inputs=[image_input, seed, num_inference_steps, guidance_scale, faces],
                    outputs=output_model
                )
            
            with gr.TabItem("Scribble to 3D"):
                with gr.Row():
                    with gr.Column():
                        scribble_input = gr.Image(type="pil", label="Input Scribble")
                        prompt = gr.Textbox(label="Prompt")
                        scribble_seed = gr.Slider(minimum=0, maximum=1000, value=42, step=1, label="Seed")
                        scribble_steps = gr.Slider(minimum=1, maximum=100, value=16, step=1, label="Inference Steps")
                        scribble_conf = gr.Slider(minimum=0.1, maximum=1.0, value=0.4, step=0.1, label="Scribble Confidence")
                        prompt_conf = gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.1, label="Prompt Confidence")
                        generate_scribble_btn = gr.Button("Generate 3D Model")
                    
                    with gr.Column():
                        scribble_output = gr.Model3D(label="Generated 3D Model")
                
                generate_scribble_btn.click(
                    fn=generate_3d_from_scribble,
                    inputs=[scribble_input, prompt, scribble_seed, scribble_steps, scribble_conf, prompt_conf],
                    outputs=scribble_output
                )
        
        gr.Markdown("""
        ## Usage Instructions
        1. **Image to 3D**: Upload an image and adjust parameters to generate a 3D model
        2. **Scribble to 3D**: Upload a scribble image, provide a prompt, and adjust parameters to generate a 3D model
        
        Note: The first run will download the model weights, which may take some time.
        """)
    
    return demo

if __name__ == "__main__":
    download_models()
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    ) 
