#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and


import argparse
import logging
import math
import os
import shutil

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from accelerate.state import AcceleratorState
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers import UNet2DConditionModel_New
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate

from loss import HOLA_LOSS_Tools
from dataset import ImageStore, AspectBucket, AspectBucketSampler, SDXL_AspectDataset

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--accelerators_save_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint_path",
        type=str,
        default=None,
        help=("Whether training should be resumed from a previous checkpoint."),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--train_data_root",
        type=str,
        default="./data/",
        help=(
            "A folder containing the training data."
        ),
    )
    parser.add_argument('--dataset', type=str, default=None, required=True, help='The path to the dataset to use for finetuning.')
    parser.add_argument('--num_buckets', type=int, default=16, help='The number of buckets.')
    parser.add_argument('--bucket_side_min', type=int, default=512, help='The minimum side length of a bucket.')
    parser.add_argument('--bucket_side_max', type=int, default=1024, help='The maximum side length of a bucket.')
    parser.add_argument('--ucg', type=float, default=0.1, help='Percentage chance of dropping out the text condition per batch. Ranges from 0.0 to 1.0 where 1.0 means 100% text condition dropout.') # 10% dropout probability
    
    parser.add_argument('--hola_loss_weight', type=float, default=1, help="The weight of hola loss.")
    parser.add_argument('--attn_map_size', type=int, default=16, help="The size of the attention map used")
    parser.add_argument("--checkpointing_save_steps", type=int, default=None)
    parser.add_argument("--num_of_parsing_categories", type=int, default=18)
    parser.add_argument("--dropout_rate_of_parsing", type=float, default=0.1)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--special_config_file", type=str, default="./configs/sdxl_unet_config.json")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        even_batches=False
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder='tokenizer_2', torch_dtype=torch.float16, variant="fp16", use_safetensors=True)

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", use_safetensors=True
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder='text_encoder_2', use_safetensors=True)
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae",  use_safetensors=True
        )
    unet = UNet2DConditionModel_New.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, "unet"),
        special_config_file=args.special_config_file,
        torch_dtype=torch.float16)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel_New.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, "unet"),
            special_config_file=args.special_config_file,
            torch_dtype=torch.float16, revision=args.revision)
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel_New, model_config=ema_unet.config, decay=args.ema_decay)

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr
    
    def _get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        passed_add_embed_dim = (
            unet.module.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = unet.module.add_embedding.linear_1.in_features
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids
    
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel_New)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel_New.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    from accelerate.utils import DummyOptim, DummyScheduler
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Initialize the dataloader
    store = ImageStore(args, logger=logger)
    train_dataset = SDXL_AspectDataset(store, tokenizer, text_encoder, tokenizer_2, text_encoder_2, accelerator.device, ucg=args.ucg, logger=logger)
    bucket = AspectBucket(store=store, num_buckets=args.num_buckets, batch_size=args.train_batch_size, 
                          bucket_side_min=args.bucket_side_min, bucket_side_max=args.bucket_side_max, bucket_side_increment=64, 
                          max_image_area=1024*1280, max_ratio=2.0)
    rank = accelerator.process_index
    sampler = AspectBucketSampler(bucket=bucket, num_replicas=accelerator.num_processes, rank=rank)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=args.dataloader_num_workers,
        collate_fn=train_dataset.collate_fn,
        batch_size=1
    )
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_batch_size

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=args.max_train_steps, warmup_num_steps=args.lr_warmup_steps
        )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        # tracker_config.pop("validation_prompt")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint_path:
        if args.resume_from_checkpoint_path is not None:
            accelerator.load_state(args.resume_from_checkpoint_path)
            global_step = int(args.resume_from_checkpoint_path.split("-")[-1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    # Define Hola Loss
    hola_loss = HOLA_LOSS_Tools(args.attn_map_size)


    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_all_loss = 0.0
        train_map_loss = 0.0
        train_noise_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint_path and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(accelerator.device, weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the embedding for conditioning
                encoder_hidden_states = batch['input_ids']
                original_size = batch['original_size']
                target_size = batch['original_size']

                crops_coords_top_left = (0, 0)
                original_size = original_size or (1024, 1024)
                target_size = target_size or (1024, 1024)

                add_text_embeds = batch['pooled_prompt_embeds']
                add_time_ids = _get_add_time_ids(
                    original_size, crops_coords_top_left, target_size, dtype=add_text_embeds.dtype
                )
                add_time_ids = add_time_ids.to(device=accelerator.device).repeat(args.train_batch_size, 1)
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                if not batch['use_parsing'] or args.hola_loss_weight == 0.:
                    unet_out = unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
                    model_pred = unet_out[0].sample
                    map_loss = torch.Tensor([0.]).to(noisy_latents.device)[0]
                else:
                    unet_out = unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs, other_info={
                        'batch': batch,
                        'loss_function': {'hola_loss': hola_loss.get_map_loss},
                    })
                    model_pred = unet_out[0].sample
                    hola_out = unet_out[1]
                    map_loss = hola_loss.merge_loss(hola_out, args.train_batch_size, noisy_latents.device) * args.hola_loss_weight

                
                if args.snr_gamma is None:
                    noise_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps)
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    noise_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    noise_loss = noise_loss.mean(dim=list(range(1, len(noise_loss.shape)))) * mse_loss_weights
                    noise_loss = noise_loss.mean()

                all_loss = noise_loss + map_loss
                # Backpropagate
                accelerator.backward(all_loss)

                # Gradient clipping after synchronizing gradients
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    # Step the optimizer and scheduler if necessary
                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()

                # Reset the gradients for the next iteration
                optimizer.zero_grad()

                # Gather and average the losses across all processes
                avg_noise_loss = accelerator.gather(noise_loss).mean()
                avg_map_loss = accelerator.gather(map_loss).mean()
                avg_all_loss = accelerator.gather(all_loss).mean()

                # Accumulate the average losses for logging purposes
                train_noise_loss += avg_noise_loss.item() / args.gradient_accumulation_steps
                train_map_loss += avg_map_loss.item() / args.gradient_accumulation_steps
                train_all_loss += avg_all_loss.item() / args.gradient_accumulation_steps

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"noise_loss": train_noise_loss}, step=global_step)
                accelerator.log({"map_loss": train_map_loss}, step=global_step)
                accelerator.log({"train_loss": train_all_loss}, step=global_step)
                train_noise_loss = 0.0
                train_map_loss = 0.0
                train_all_loss = 0.0

                if accelerator.is_main_process:
                    # save checkpoints
                    if global_step % args.checkpointing_save_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        unet.save_pretrained(os.path.join(save_path, "unet"))
                        logger.info(f"Saved weights to {save_path}")

                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                if global_step % args.accelerators_save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            logs = {"lr": lr_scheduler.get_last_lr()[0]}
            if noise_loss.detach().item() > 0.:
                logs['step_noise_loss'] = noise_loss.detach().item()
            if all_loss.detach().item() > 0.:
                logs['step_all_loss'] = all_loss.detach().item()
            if map_loss.detach().item() > 0.:
                logs['step_map_loss'] = map_loss.detach().item()
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
            


    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()

