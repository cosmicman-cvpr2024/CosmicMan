# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
# limitations under the License.

import copy
import random
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    StableDiffusionXLInpaintPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.utils import floats_tensor, torch_device
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_gpu

from ..pipeline_params import TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS, TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
from ..test_pipelines_common import PipelineLatentTesterMixin, PipelineTesterMixin


enable_full_determinism()


class StableDiffusionXLInpaintPipelineFastTests(PipelineLatentTesterMixin, PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionXLInpaintPipeline
    params = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    image_params = frozenset([])
    # TO-DO: update image_params once pipeline is refactored with VaeImageProcessor.preprocess
    image_latents_params = frozenset([])

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
        )
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            # SD2-specific config below
            hidden_act="gelu",
            projection_dim=32,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip", local_files_only=True)

        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip", local_files_only=True)

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        # TODO: use tensor inputs instead of PIL, this is here just to leave the old expected_slices untouched
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        image = image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((64, 64))
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_xl_inpaint_euler(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLInpaintPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array([0.4924, 0.4966, 0.4100, 0.5233, 0.5322, 0.4532, 0.5804, 0.5876, 0.4150])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_attention_slicing_forward_pass(self):
        super().test_attention_slicing_forward_pass(expected_max_diff=3e-3)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)

    # TODO(Patrick, Sayak) - skip for now as this requires more refiner tests
    def test_save_load_optional_components(self):
        pass

    def test_stable_diffusion_xl_inpaint_negative_prompt_embeds(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLInpaintPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        # forward without prompt embeds
        inputs = self.get_dummy_inputs(torch_device)
        negative_prompt = 3 * ["this is a negative prompt"]
        inputs["negative_prompt"] = negative_prompt
        inputs["prompt"] = 3 * [inputs["prompt"]]

        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        # forward with prompt embeds
        inputs = self.get_dummy_inputs(torch_device)
        negative_prompt = 3 * ["this is a negative prompt"]
        prompt = 3 * [inputs.pop("prompt")]

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = sd_pipe.encode_prompt(prompt, negative_prompt=negative_prompt)

        output = sd_pipe(
            **inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )
        image_slice_2 = output.images[0, -3:, -3:, -1]

        # make sure that it's equal
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

    @require_torch_gpu
    def test_stable_diffusion_xl_offloads(self):
        pipes = []
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLInpaintPipeline(**components).to(torch_device)
        pipes.append(sd_pipe)

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLInpaintPipeline(**components)
        sd_pipe.enable_model_cpu_offload()
        pipes.append(sd_pipe)

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLInpaintPipeline(**components)
        sd_pipe.enable_sequential_cpu_offload()
        pipes.append(sd_pipe)

        image_slices = []
        for pipe in pipes:
            pipe.unet.set_default_attn_processor()

            inputs = self.get_dummy_inputs(torch_device)
            image = pipe(**inputs).images

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert np.abs(image_slices[0] - image_slices[1]).max() < 1e-3
        assert np.abs(image_slices[0] - image_slices[2]).max() < 1e-3

    def test_stable_diffusion_two_xl_mixture_of_denoiser(self):
        components = self.get_dummy_components()
        pipe_1 = StableDiffusionXLInpaintPipeline(**components).to(torch_device)
        pipe_1.unet.set_default_attn_processor()
        pipe_2 = StableDiffusionXLInpaintPipeline(**components).to(torch_device)
        pipe_2.unet.set_default_attn_processor()

        def assert_run_mixture(num_steps, split, scheduler_cls_orig):
            inputs = self.get_dummy_inputs(torch_device)
            inputs["num_inference_steps"] = num_steps

            class scheduler_cls(scheduler_cls_orig):
                pass

            pipe_1.scheduler = scheduler_cls.from_config(pipe_1.scheduler.config)
            pipe_2.scheduler = scheduler_cls.from_config(pipe_2.scheduler.config)

            # Let's retrieve the number of timesteps we want to use
            pipe_1.scheduler.set_timesteps(num_steps)
            expected_steps = pipe_1.scheduler.timesteps.tolist()

            split_id = int(round(split * num_steps)) * pipe_1.scheduler.order
            expected_steps_1 = expected_steps[:split_id]
            expected_steps_2 = expected_steps[split_id:]

            # now we monkey patch step `done_steps`
            # list into the step function for testing
            done_steps = []
            old_step = copy.copy(scheduler_cls.step)

            def new_step(self, *args, **kwargs):
                done_steps.append(args[1].cpu().item())  # args[1] is always the passed `t`
                return old_step(self, *args, **kwargs)

            scheduler_cls.step = new_step

            inputs_1 = {**inputs, **{"denoising_end": split, "output_type": "latent"}}
            latents = pipe_1(**inputs_1).images[0]

            assert expected_steps_1 == done_steps, f"Failure with {scheduler_cls.__name__} and {num_steps} and {split}"

            inputs_2 = {**inputs, **{"denoising_start": split, "image": latents}}
            pipe_2(**inputs_2).images[0]

            assert expected_steps_2 == done_steps[len(expected_steps_1) :]
            assert expected_steps == done_steps, f"Failure with {scheduler_cls.__name__} and {num_steps} and {split}"

        for steps in [5, 8]:
            for split in [0.33, 0.49, 0.71]:
                for scheduler_cls in [
                    DDIMScheduler,
                    EulerDiscreteScheduler,
                    DPMSolverMultistepScheduler,
                    UniPCMultistepScheduler,
                    HeunDiscreteScheduler,
                ]:
                    assert_run_mixture(steps, split, scheduler_cls)

    def test_stable_diffusion_three_xl_mixture_of_denoiser(self):
        components = self.get_dummy_components()
        pipe_1 = StableDiffusionXLInpaintPipeline(**components).to(torch_device)
        pipe_1.unet.set_default_attn_processor()
        pipe_2 = StableDiffusionXLInpaintPipeline(**components).to(torch_device)
        pipe_2.unet.set_default_attn_processor()
        pipe_3 = StableDiffusionXLInpaintPipeline(**components).to(torch_device)
        pipe_3.unet.set_default_attn_processor()

        def assert_run_mixture(num_steps, split_1, split_2, scheduler_cls_orig):
            inputs = self.get_dummy_inputs(torch_device)
            inputs["num_inference_steps"] = num_steps

            class scheduler_cls(scheduler_cls_orig):
                pass

            pipe_1.scheduler = scheduler_cls.from_config(pipe_1.scheduler.config)
            pipe_2.scheduler = scheduler_cls.from_config(pipe_2.scheduler.config)
            pipe_3.scheduler = scheduler_cls.from_config(pipe_3.scheduler.config)

            # Let's retrieve the number of timesteps we want to use
            pipe_1.scheduler.set_timesteps(num_steps)
            expected_steps = pipe_1.scheduler.timesteps.tolist()

            split_id_1 = int(round(split_1 * num_steps)) * pipe_1.scheduler.order
            split_id_2 = int(round(split_2 * num_steps)) * pipe_1.scheduler.order
            expected_steps_1 = expected_steps[:split_id_1]
            expected_steps_2 = expected_steps[split_id_1:split_id_2]
            expected_steps_3 = expected_steps[split_id_2:]

            # now we monkey patch step `done_steps`
            # list into the step function for testing
            done_steps = []
            old_step = copy.copy(scheduler_cls.step)

            def new_step(self, *args, **kwargs):
                done_steps.append(args[1].cpu().item())  # args[1] is always the passed `t`
                return old_step(self, *args, **kwargs)

            scheduler_cls.step = new_step

            inputs_1 = {**inputs, **{"denoising_end": split_1, "output_type": "latent"}}
            latents = pipe_1(**inputs_1).images[0]

            assert (
                expected_steps_1 == done_steps
            ), f"Failure with {scheduler_cls.__name__} and {num_steps} and {split_1} and {split_2}"

            inputs_2 = {
                **inputs,
                **{"denoising_start": split_1, "denoising_end": split_2, "image": latents, "output_type": "latent"},
            }
            pipe_2(**inputs_2).images[0]

            assert expected_steps_2 == done_steps[len(expected_steps_1) :]

            inputs_3 = {**inputs, **{"denoising_start": split_2, "image": latents}}
            pipe_3(**inputs_3).images[0]

            assert expected_steps_3 == done_steps[len(expected_steps_1) + len(expected_steps_2) :]
            assert (
                expected_steps == done_steps
            ), f"Failure with {scheduler_cls.__name__} and {num_steps} and {split_1} and {split_2}"

        for steps in [7, 11]:
            for split_1, split_2 in zip([0.19, 0.32], [0.81, 0.68]):
                for scheduler_cls in [
                    DDIMScheduler,
                    EulerDiscreteScheduler,
                    DPMSolverMultistepScheduler,
                    UniPCMultistepScheduler,
                    HeunDiscreteScheduler,
                ]:
                    assert_run_mixture(steps, split_1, split_2, scheduler_cls)
