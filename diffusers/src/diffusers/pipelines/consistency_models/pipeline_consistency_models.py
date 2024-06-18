from typing import Callable, List, Optional, Union

import torch

from ...models import UNet2DModel
from ...schedulers import CMStochasticIterativeScheduler
from ...utils import (
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch

        >>> from diffusers import ConsistencyModelPipeline

        >>> device = "cuda"
        >>> # Load the cd_imagenet64_l2 checkpoint.
        >>> model_id_or_path = "openai/diffusers-cd_imagenet64_l2"
        >>> pipe = ConsistencyModelPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
        >>> pipe.to(device)

        >>> # Onestep Sampling
        >>> image = pipe(num_inference_steps=1).images[0]
        >>> image.save("cd_imagenet64_l2_onestep_sample.png")

        >>> # Onestep sampling, class-conditional image generation
        >>> # ImageNet-64 class label 145 corresponds to king penguins
        >>> image = pipe(num_inference_steps=1, class_labels=145).images[0]
        >>> image.save("cd_imagenet64_l2_onestep_sample_penguin.png")

        >>> # Multistep sampling, class-conditional image generation
        >>> # Timesteps can be explicitly specified; the particular timesteps below are from the original Github repo:
        >>> # https://github.com/openai/consistency_models/blob/main/scripts/launch.sh#L77
        >>> image = pipe(num_inference_steps=None, timesteps=[22, 0], class_labels=145).images[0]
        >>> image.save("cd_imagenet64_l2_multistep_sample_penguin.png")
        ```
"""


class ConsistencyModelPipeline(DiffusionPipeline):
    r"""
    Pipeline for consistency models for unconditional or class-conditional image generation, as introduced in [1].

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    [1] Song, Yang and Dhariwal, Prafulla and Chen, Mark and Sutskever, Ilya. "Consistency Models"
    https://arxiv.org/pdf/2303.01469

    Args:
        unet ([`UNet2DModel`]):
            Unconditional or class-conditional U-Net architecture to denoise image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the image latents. Currently only compatible
            with [`CMStochasticIterativeScheduler`].
    """

    def __init__(self, unet: UNet2DModel, scheduler: CMStochasticIterativeScheduler) -> None:
        super().__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
        )

        self.safety_checker = None

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.unet]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        if self.safety_checker is not None:
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    def prepare_latents(self, batch_size, num_channels, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Follows diffusers.VaeImageProcessor.postprocess
    def postprocess_image(self, sample: torch.FloatTensor, output_type: str = "pil"):
        if output_type not in ["pt", "np", "pil"]:
            raise ValueError(
                f"output_type={output_type} is not supported. Make sure to choose one of ['pt', 'np', or 'pil']"
            )

        # Equivalent to diffusers.VaeImageProcessor.denormalize
        sample = (sample / 2 + 0.5).clamp(0, 1)
        if output_type == "pt":
            return sample

        # Equivalent to diffusers.VaeImageProcessor.pt_to_numpy
        sample = sample.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "np":
            return sample

        # Output_type must be 'pil'
        sample = self.numpy_to_pil(sample)
        return sample

    def prepare_class_labels(self, batch_size, device, class_labels=None):
        if self.unet.config.num_class_embeds is not None:
            if isinstance(class_labels, list):
                class_labels = torch.tensor(class_labels, dtype=torch.int)
            elif isinstance(class_labels, int):
                assert batch_size == 1, "Batch size must be 1 if classes is an int"
                class_labels = torch.tensor([class_labels], dtype=torch.int)
            elif class_labels is None:
                # Randomly generate batch_size class labels
                # TODO: should use generator here? int analogue of randn_tensor is not exposed in ...utils
                class_labels = torch.randint(0, self.unet.config.num_class_embeds, size=(batch_size,))
            class_labels = class_labels.to(device)
        else:
            class_labels = None
        return class_labels

    def check_inputs(self, num_inference_steps, timesteps, latents, batch_size, img_size, callback_steps):
        if num_inference_steps is None and timesteps is None:
            raise ValueError("Exactly one of `num_inference_steps` or `timesteps` must be supplied.")

        if num_inference_steps is not None and timesteps is not None:
            logger.warning(
                f"Both `num_inference_steps`: {num_inference_steps} and `timesteps`: {timesteps} are supplied;"
                " `timesteps` will be used over `num_inference_steps`."
            )

        if latents is not None:
            expected_shape = (batch_size, 3, img_size, img_size)
            if latents.shape != expected_shape:
                raise ValueError(f"The shape of latents is {latents.shape} but is expected to be {expected_shape}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        batch_size: int = 1,
        class_labels: Optional[Union[torch.Tensor, List[int], int]] = None,
        num_inference_steps: int = 1,
        timesteps: List[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
    ):
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            class_labels (`torch.Tensor` or `List[int]` or `int`, *optional*):
                Optional class labels for conditioning class-conditional consistency models. Will not be used if the
                model is not class-conditional.
            num_inference_steps (`int`, *optional*, defaults to 1):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        # 0. Prepare call parameters
        img_size = self.unet.config.sample_size
        device = self._execution_device

        # 1. Check inputs
        self.check_inputs(num_inference_steps, timesteps, latents, batch_size, img_size, callback_steps)

        # 2. Prepare image latents
        # Sample image latents x_0 ~ N(0, sigma_0^2 * I)
        sample = self.prepare_latents(
            batch_size=batch_size,
            num_channels=self.unet.config.in_channels,
            height=img_size,
            width=img_size,
            dtype=self.unet.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 3. Handle class_labels for class-conditional models
        class_labels = self.prepare_class_labels(batch_size, device, class_labels=class_labels)

        # 4. Prepare timesteps
        if timesteps is not None:
            self.scheduler.set_timesteps(timesteps=timesteps, device=device)
            timesteps = self.scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps

        # 5. Denoising loop
        # Multistep sampling: implements Algorithm 1 in the paper
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                scaled_sample = self.scheduler.scale_model_input(sample, t)
                model_output = self.unet(scaled_sample, t, class_labels=class_labels, return_dict=False)[0]

                sample = self.scheduler.step(model_output, t, sample, generator=generator)[0]

                # call the callback, if provided
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, sample)

        # 6. Post-process image sample
        image = self.postprocess_image(sample, output_type=output_type)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
