
import gradio as gr
from PIL import Image
import torch
from diffusers import Flux2KleinPipeline
import re
import os

os.makedirs("./outputs", exist_ok=True)

def to_file_safe(s: str, max_length: int = 255) -> str:
    s = s.strip().lower()
    s = re.sub(r'[^\w\-\.]', '_', s)
    s = re.sub(r'_+', '_', s)
    return s[:max_length]

# Model setup
device = "cuda"
dtype = torch.bfloat16
# Ensure this custom pipeline class is correctly imported/available
pipe = Flux2KleinPipeline.from_pretrained("./", torch_dtype=dtype)
pipe.enable_model_cpu_offload()

def generate_ai_image(prompt, pil_images, guidance, steps, width, height):
    # Handle empty image list safely
    ref_images = pil_images if len(pil_images) > 0 else None

    # Run the generation pipeline
    # Note: Check if Flux2KleinPipeline supports a list of images or just one
    ai_image = pipe(
        prompt=prompt,
        image=ref_images,
        height=height,
        width=width,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=torch.Generator(device=device).manual_seed(0)
    ).images[0]

    save_path = f"./outputs/{to_file_safe(prompt, 60)}.png"
    ai_image.save(save_path)
    return ai_image

def process_interface(prompt, image_files, guidance, steps, width, height):
    pil_images = []
    if image_files:
        for file_path in image_files:
            # image_files is a list of file paths (strings) in Gradio 4+
            img = Image.open(file_path)
            pil_images.append(img.convert("RGB")) # Ensure RGB mode

    return generate_ai_image(prompt, pil_images, guidance, steps, width, height)

with gr.Blocks() as demo:
    gr.Markdown("# FLUX.2 Klein Image Generator")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Image Prompt", placeholder="Describe what you want...")
            image_input = gr.File(label="Upload Reference Images", file_count="multiple", file_types=["image"])

            with gr.Accordion("Advanced Settings", open=True):
                guidance_slider = gr.Slider(minimum=1.0, maximum=20.0, value=4.0, step=0.5, label="Guidance Scale")
                steps_slider = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Steps")
                # Changed step to 8 for tensor compatibility
                width_slider = gr.Slider(minimum=256, maximum=1920, value=1024, step=8, label="Width")
                height_slider = gr.Slider(minimum=256, maximum=1080, value=1024, step=8, label="Height")

            generate_btn = gr.Button("Generate Image", variant="primary")

        with gr.Column():
            image_output = gr.Image(label="AI Generated Result")

    generate_btn.click(
        fn=process_interface,
        inputs=[prompt_input, image_input, guidance_slider, steps_slider, width_slider, height_slider],
        outputs=image_output
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)
