from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, UNet2DConditionModel
from diffusers.utils import load_image
from diffusers import (
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
import torch
import os
import random
import numpy as np
from PIL import Image
from typing import Tuple
import gradio as gr
DESCRIPTION =  """
    # CosmicMan
    - CosmicMan: A Text-to-Image Foundation Model for Humans (CVPR 2024 (Highlight))
    """

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

schedule_map = {
    "ddim"   : DDIMScheduler,
    "pndm"   : PNDMScheduler,
    "lms"    : LMSDiscreteScheduler,
    "euler"  : EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "dpm"    : DPMSolverMultistepScheduler,
}

examples = [
    "A fit Caucasian elderly woman, her wavy white hair above shoulders, wears a pink floral cotton long-sleeve shirt and a cotton hat against a natural landscape in an upper body shot",
    "A closeup of a doll with a purple ribbon around her neck, best quality, extremely detailed",
    "A closeup of a girl with a butterfly painted on her face",
    "A headshot, an asian elderly male, a blue wall, bald above eyes gray hair",
    "A closeup portrait shot against a white wall, a fit Caucasian adult female with wavy blonde hair falling above her chest wears a short sleeve silk floral dress and a floral silk normal short sleeve white blouse",
    "A headshot, an adult caucasian male, fit, a white wall, red crew cut curly hair, short sleeve normal blue t-shirt, best quality, extremely detailed",
    "A closeup of a man wearing a red shirt with a flower design on it",
    "There is a man wearing a mask and holding a cell phone",
    "Two boys playing in the yard",
]

style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    }
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "1") == "1"
MAX_SEED = np.iinfo(np.int32).max
NUM_IMAGES_PER_PROMPT = 1 
 
def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative

class NoWatermark:
    def apply_watermark(self, img):
        return img
    
def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

print("Loading Model!")
schedule: str = "euler_a"
base_model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
refiner_model_path: str = "stabilityai/stable-diffusion-xl-refiner-1.0" 
unet_path: str = "cosmicman/CosmicMan-SDXL"
SCHEDULER = schedule_map[schedule]
scheduler = SCHEDULER.from_pretrained(base_model_path, subfolder="scheduler", torch_dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16)

pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    unet=unet,
    scheduler=scheduler,
    torch_dtype=torch.float16, 
    use_safetensors=True
).to("cuda")
pipe.watermark = NoWatermark()
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    base_model_path, # we found use base_model_path instead of refiner_model_path may get a better performance
    scheduler=scheduler,
    torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
refiner.watermark = NoWatermark()
print("Finish Loading Model!")

def generate_image(prompt, 
                   n_prompt="", 
                   style: str = DEFAULT_STYLE_NAME,
                   steps: int = 50, 
                   height: int = 1024,
                   width: int = 1024,
                   scale: float = 7.5, 
                   img_num: int = 4, 
                   seeds: int = 42,
                   random_seed: bool = False,
):
    print("Beign to generate")
    image_list = []
    for i in range(img_num):
        generator = torch.Generator(device="cuda")
        seed = int(randomize_seed_fn(seeds, random_seed))
        generator = torch.Generator().manual_seed(seed)
        positive_prompt, negative_prompt = apply_style(style, prompt, n_prompt)
        image = pipe(positive_prompt, num_inference_steps=steps, 
                guidance_scale=scale, height=height, 
                width=width, negative_prompt=negative_prompt, 
                generator=generator, output_type="latent").images[0]
        image = refiner(positive_prompt, negative_prompt=negative_prompt, image=image[None, :]).images[0]
        image_list.append((image,f"Seed {seed}"))
    return image_list

with gr.Blocks(theme=gr.themes.Soft(),css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Group():
        with gr.Row():
            with gr.Column():
                input_prompt = gr.Textbox(label="Input prompt", lines=3, max_lines=5)
                negative_prompt = gr.Textbox(label="Negative prompt",value="")
                run_button = gr.Button("Run", scale=0)
        result = gr.Gallery(label="Result", show_label=False, elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto")
    with gr.Accordion("Advanced options", open=False):
        with gr.Row():
            style_selection = gr.Radio(
                    show_label=True,
                    container=True,
                    interactive=True,
                    choices=STYLE_NAMES,
                    value=DEFAULT_STYLE_NAME,
                    label="Image Style",
                )
        with gr.Row():       
            height = gr.Slider(minimum=512, maximum=1536, value=1024, label="Height", step=64)
            width = gr.Slider(minimum=512, maximum=1536, value=1024, label="Witdh", step=64) 
        with gr.Row():
            steps =  gr.Slider(minimum=1, maximum=50, value=30, label="Number of diffusion steps", step=1)
            scale = gr.Number(minimum=1, maximum=12, value=7.5, label="Number of scale")        
        with gr.Row():     
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            random_seed = gr.Checkbox(label="Randomize seed", value=True)
            img_num = gr.Slider(minimum=1, maximum=4, value=4, label="Number of images", step=1)   

    gr.Examples(
        examples=examples,
        inputs=input_prompt,
        outputs=result,
        fn=generate_image,
        cache_examples=CACHE_EXAMPLES,
    )

    gr.on(
        triggers=[
            input_prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=generate_image, 
        inputs = [input_prompt, negative_prompt, style_selection, steps, height, width, scale, img_num, seed, random_seed],
        outputs= result,
        api_name="run")


if __name__ == "__main__":
    demo.queue(max_size=20)
    demo.launch(share=True, server_name='0.0.0.0', server_port=10057)
    