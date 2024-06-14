import os
import argparse
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers import (
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

sampler_map = {
    "ddim"   : DDIMScheduler,
    "pndm"   : PNDMScheduler,
    "lms"    : LMSDiscreteScheduler,
    "euler"  : EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "dpm"    : DPMSolverMultistepScheduler,

}

class NoWatermark:
    def apply_watermark(self, img):
        return img
    
parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to", default="tmp/test")
parser.add_argument("--unet_path", type=str, default="cosmicman/CosmicMan-SD")
parser.add_argument("--steps", type=int, default=50, help="number of ddim sampling steps",)
parser.add_argument("--H", type=int, default=768, help="image height, in pixel space",)
parser.add_argument("--W", type=int, default=768, help="image width, in pixel space",)
parser.add_argument("--scale", type=float, default=7.5, help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",)
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--n_prompt", type=str, default='')
parser.add_argument('--a_prompt', type=str, default='')
parser.add_argument('--sampler', type=str, default="ddim",)
parser.add_argument("--model_path", type=str, default="runwayml/stable-diffusion-v1-5")
parser.add_argument("--prompts",type=str, default=None, nargs="+")
args = parser.parse_args()
outpath = args.outdir

if not os.path.exists(outpath):
    os.makedirs(outpath)

print("Loading model...")
SCHEDULER = sampler_map[args.sampler]
scheduler = SCHEDULER.from_pretrained(args.model_path, subfolder="scheduler", torch_dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(args.unet_path, torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    args.model_path,
    unet=unet,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    variant="fp16", use_safetensors=True
).to("cuda")
pipe.watermark = NoWatermark()
generator = torch.Generator(device="cuda")
generator = generator.manual_seed(args.seed)
print("init model done")

for i, prompt in enumerate(args.prompts):
    prompt_a = prompt + ", " + args.a_prompt
    image = pipe(prompt_a, num_inference_steps=args.steps, 
                guidance_scale=args.scale, height=args.H, 
                width=args.W, negative_prompt=args.n_prompt, 
                generator=generator, output_type="pil").images[0]
    prefix = str(i).rjust(4,'0')
    image.save(os.path.join(outpath, prefix +'_' + f'{prompt[:128].replace(" ", "-").replace("/", "-")}.png'))
 