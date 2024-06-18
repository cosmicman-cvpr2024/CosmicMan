# DreamBooth training example for Stable Diffusion XL (SDXL)

[DreamBooth](https://arxiv.org/abs/2208.12242) is a method to personalize text2image models like stable diffusion given just a few (3~5) images of a subject.

The `train_dreambooth_lora_sdxl.py` script shows how to implement the training procedure and adapt it for [Stable Diffusion XL](https://huggingface.co/papers/2307.01952).

> 💡 **Note**: For now, we only allow DreamBooth fine-tuning of the SDXL UNet via LoRA. LoRA is a parameter-efficient fine-tuning technique introduced in [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen*. 

## Running locally with PyTorch

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then cd in the `examples/dreambooth` folder and run
```bash
pip install -r requirements_sdxl.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell (e.g., a notebook)

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

When running `accelerate config`, if we specify torch compile mode to True there can be dramatic speedups. 

### Dog toy example

Now let's get our dataset. For this example we will use some dog images: https://huggingface.co/datasets/diffusers/dog-example.

Let's first download it locally:

```python
from huggingface_hub import snapshot_download

local_dir = "./dog"
snapshot_download(
    "diffusers/dog-example",
    local_dir=local_dir, repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```

Since SDXL 0.9 weights are gated, we need to be authenticated to be able to use them. So, let's run:

```bash
huggingface-cli login
```

This will also allow us to push the trained LoRA parameters to the Hugging Face Hub platform. 

Now, we can launch training using:

```bash
export MODEL_NAME="diffusers/stable-diffusion-xl-base-0.9"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="lora-trained-xl"

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0" \
  --push_to_hub
```

To better track our training experiments, we're using the following flags in the command above:

* `report_to="wandb` will ensure the training runs are tracked on Weights and Biases. To use it, be sure to install `wandb` with `pip install wandb`.
* `validation_prompt` and `validation_epochs` to allow the script to do a few validation inference runs. This allows us to qualitatively check if the training is progressing as expected. 

Our experiments were conducted on a single 40GB A100 GPU.

### Inference

Once training is done, we can perform inference like so:

```python
from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline
import torch

lora_model_id = <"lora-sdxl-dreambooth-id">
card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.load_lora_weights(lora_model_id)
image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
image.save("sks_dog.png")
```

We can further refine the outputs with the [Refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-0.9):

```python
from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch

lora_model_id = <"lora-sdxl-dreambooth-id">
card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

# Load the base pipeline and load the LoRA parameters into it. 
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.load_lora_weights(lora_model_id)

# Load the refiner.
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
)
refiner.to("cuda")

prompt = "A picture of a sks dog in a bucket"
generator = torch.Generator("cuda").manual_seed(0)

# Run inference.
image = pipe(prompt=prompt, output_type="latent", generator=generator).images[0]
image = refiner(prompt=prompt, image=image[None, :], generator=generator).images[0]
image.save("refined_sks_dog.png")
```

Here's a side-by-side comparison of the with and without Refiner pipeline outputs:

| Without Refiner | With Refiner |
|---|---|
| ![](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/sd_xl/sks_dog.png) | ![](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/sd_xl/refined_sks_dog.png) |

## Notes

In our experiments we found that SDXL yields very good initial results using the default settings of the script. We didn't explore further hyper-parameter tuning experiments, but we do encourage the community to explore this avenue further and share their results with us 🤗

## Results

You can explore the results from a couple of our internal experiments by checking out this link: [https://wandb.ai/sayakpaul/dreambooth-lora-sd-xl](https://wandb.ai/sayakpaul/dreambooth-lora-sd-xl). Specifically, we used the same script with the exact same hyperparameters on the following datasets:

* [Dogs](https://huggingface.co/datasets/diffusers/dog-example)
* [Starbucks logo](https://huggingface.co/datasets/diffusers/starbucks-example)
* [Mr. Potato Head](https://huggingface.co/datasets/diffusers/potato-head-example)
* [Keramer face](https://huggingface.co/datasets/diffusers/keramer-face-example)