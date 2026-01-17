import gradio as gr
from PIL import Image
import torch
from diffusers import Flux2KleinPipeline
import re
import os

# Ensure the output directory exists
os.makedirs("./outputs", exist_ok=True)

# Helper for filename safety
def to_file_safe(s: str, max_length: int = 255) -> str:
    s = s.strip().lower()
    s = re.sub(r'[^\w\-\.]', '_', s)
    s = re.sub(r'_+', '_', s)
    return s[:max_length]

# Model setup
device = "cuda"
dtype = torch.bfloat16
pipe = Flux2KleinPipeline.from_pretrained("./", torch_dtype=dtype)
pipe.enable_model_cpu_offload()

def generate_ai_image(prompt, images, guidance, steps):
    """
    Takes prompt, images, and slider values to generate an image.
    """
    # Use the first image if available, otherwise handle empty input
    ref_image = images[0] if images else None
    
    # Run the generation pipeline
    ai_image = pipe(
        prompt=prompt,
        image=images,#ref_image,
        height=1024,
        width=1024,
        guidance_scale=guidance, # Linked to slider
        num_inference_steps=steps, # Linked to slider
        generator=torch.Generator(device=device).manual_seed(0)
    ).images[0]

    # Save to disk
    ai_image.save(f"./outputs/{to_file_safe(prompt)}.png")
    return ai_image

def process_interface(prompt, image_files, guidance, steps):
    """
    Converts Gradio files to PIL and passes UI parameters to the generator.
    """
    pil_images = []
    if image_files:
        for file in image_files:
            img = Image.open(file.name)
            pil_images.append(img)

    # Pass the sliders' values down to the model function
    return generate_ai_image(prompt, pil_images, guidance, steps)

# Building the Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# FLUX.2 Klein (Unofficial) Image Generator")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Image Prompt",
                placeholder="Describe what you want to see..."
            )
            
            image_input = gr.File(
                label="Upload Reference Images",
                file_count="multiple",
                file_types=["image"]
            )

            # --- New UI Options ---
            with gr.Accordion("Advanced Settings", open=True):
                guidance_slider = gr.Slider(
                    minimum=1.0, 
                    maximum=20.0, 
                    value=4.0, 
                    step=0.5, 
                    label="Guidance Scale"
                )
                steps_slider = gr.Slider(
                    minimum=1, 
                    maximum=50, 
                    value=10, 
                    step=1, 
                    label="Number of Inference Steps"
                )
            
            generate_btn = gr.Button("Generate Image", variant="primary")

        with gr.Column():
            image_output = gr.Image(label="AI Generated Result")

    # Connect the inputs (including sliders) to the function
    generate_btn.click(
        fn=process_interface,
        inputs=[prompt_input, image_input, guidance_slider, steps_slider],
        outputs=image_output
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)
