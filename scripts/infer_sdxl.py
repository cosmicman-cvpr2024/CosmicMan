import os
import argparse
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, StableDiffusionXLImg2ImgPipeline
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
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, nargs="?", default="test_results")
    parser.add_argument("--unet_path", type=str, default="cosmicman/CosmicMan-SDXL")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--H", type=int, default=1024)
    parser.add_argument("--W", type=int, default=1024)
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--n_prompt", type=str, default='')
    parser.add_argument('--a_prompt', type=str, default='') 
    parser.add_argument('--use_refiner', action='store_true')
    parser.add_argument('--sampler', type=str, default="euler_a")
    parser.add_argument("--base_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--refiner_path", type=str, default="stabilityai/stable-diffusion-xl-refiner-1.0")
    parser.add_argument("--prompts",type=str, default=None, nargs="+")
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    print("Loading model...")
    SCHEDULER = sampler_map[args.sampler]
    scheduler = SCHEDULER.from_pretrained(args.base_path, subfolder="scheduler", torch_dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(args.unet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_path,
        unet=unet,
        scheduler=scheduler,
        torch_dtype=torch.float16, 
        use_safetensors=True
    ).to("cuda")
    pipe.watermark = NoWatermark()
    if args.use_refiner:
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            args.refiner_path, # we found use base_path instead of refiner_path may get a better performance
            torch_dtype=torch.float16, use_safetensors=True
        ).to("cuda")
        refiner.watermark = NoWatermark()
    generator = torch.Generator(device="cuda")
    generator = generator.manual_seed(args.seed)
    print("init unet done")
    
    for i, prompt in enumerate(args.prompts):
        prompt_a = prompt + ", " + args.a_prompt
        image = pipe(prompt_a, num_inference_steps=args.steps, 
                    guidance_scale=args.scale, height=args.H, 
                    width=args.W, negative_prompt=args.n_prompt, 
                    generator=generator, output_type="pil" if not args.use_refiner else "latent").images[0]
        if args.use_refiner:
            image = refiner(prompt_a, negative_prompt=args.n_prompt, image=image[None, :]).images[0]
        prefix = str(i).rjust(4,'0')
        image.save(os.path.join(args.outdir, prefix +'_' + f'{prompt[:128].replace(" ", "-")}'+'.png'))
