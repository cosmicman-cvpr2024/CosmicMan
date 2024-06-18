# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, logging, randn_tensor
from .scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class CMStochasticIterativeSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class CMStochasticIterativeScheduler(SchedulerMixin, ConfigMixin):
    """
    Multistep and onestep sampling for consistency models from Song et al. 2023 [1]. This implements Algorithm 1 in the
    paper [1].

    [1] Song, Yang and Dhariwal, Prafulla and Chen, Mark and Sutskever, Ilya. "Consistency Models"
    https://arxiv.org/pdf/2303.01469 [2] Karras, Tero, et al. "Elucidating the Design Space of Diffusion-Based
    Generative Models." https://arxiv.org/abs/2206.00364

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        sigma_min (`float`):
            Minimum noise magnitude in the sigma schedule. This was set to 0.002 in the original implementation.
        sigma_max (`float`):
            Maximum noise magnitude in the sigma schedule. This was set to 80.0 in the original implementation.
        sigma_data (`float`):
            The standard deviation of the data distribution, following the EDM paper [2]. This was set to 0.5 in the
            original implementation, which is also the original value suggested in the EDM paper.
        s_noise (`float`):
            The amount of additional noise to counteract loss of detail during sampling. A reasonable range is [1.000,
            1.011]. This was set to 1.0 in the original implementation.
        rho (`float`):
            The rho parameter used for calculating the Karras sigma schedule, introduced in the EDM paper [2]. This was
            set to 7.0 in the original implementation, which is also the original value suggested in the EDM paper.
        clip_denoised (`bool`):
            Whether to clip the denoised outputs to `(-1, 1)`. Defaults to `True`.
        timesteps (`List` or `np.ndarray` or `torch.Tensor`, *optional*):
            Optionally, an explicit timestep schedule can be specified. The timesteps are expected to be in increasing
            order.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 40,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        s_noise: float = 1.0,
        rho: float = 7.0,
        clip_denoised: bool = True,
    ):
        # standard deviation of the initial noise distribution
        self.init_noise_sigma = sigma_max

        ramp = np.linspace(0, 1, num_train_timesteps)
        sigmas = self._convert_to_karras(ramp)
        timesteps = self.sigma_to_t(sigmas)

        # setable values
        self.num_inference_steps = None
        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps)
        self.custom_timesteps = False
        self.is_scale_input_called = False

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()
        return indices.item()

    def scale_model_input(
        self, sample: torch.FloatTensor, timestep: Union[float, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """
        Scales the consistency model input by `(sigma**2 + sigma_data**2) ** 0.5`, following the EDM model.

        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`float` or `torch.FloatTensor`): the current timestep in the diffusion chain
        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        # Get sigma corresponding to timestep
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        step_idx = self.index_for_timestep(timestep)
        sigma = self.sigmas[step_idx]

        sample = sample / ((sigma**2 + self.config.sigma_data**2) ** 0.5)

        self.is_scale_input_called = True
        return sample

    def sigma_to_t(self, sigmas: Union[float, np.ndarray]):
        """
        Gets scaled timesteps from the Karras sigmas, for input to the consistency model.

        Args:
            sigmas (`float` or `np.ndarray`): single Karras sigma or array of Karras sigmas
        Returns:
            `float` or `np.ndarray`: scaled input timestep or scaled input timestep array
        """
        if not isinstance(sigmas, np.ndarray):
            sigmas = np.array(sigmas, dtype=np.float64)

        timesteps = 1000 * 0.25 * np.log(sigmas + 1e-44)

        return timesteps

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
    ):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, optional):
                the device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, optional):
                custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps is used. If passed, `num_inference_steps`
                must be `None`.
        """
        if num_inference_steps is None and timesteps is None:
            raise ValueError("Exactly one of `num_inference_steps` or `timesteps` must be supplied.")

        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `timesteps`.")

        # Follow DDPMScheduler custom timesteps logic
        if timesteps is not None:
            for i in range(1, len(timesteps)):
                if timesteps[i] >= timesteps[i - 1]:
                    raise ValueError("`timesteps` must be in descending order.")

            if timesteps[0] >= self.config.num_train_timesteps:
                raise ValueError(
                    f"`timesteps` must start before `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps}."
                )

            timesteps = np.array(timesteps, dtype=np.int64)
            self.custom_timesteps = True
        else:
            if num_inference_steps > self.config.num_train_timesteps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                    f" maximal {self.config.num_train_timesteps} timesteps."
                )

            self.num_inference_steps = num_inference_steps

            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            self.custom_timesteps = False

        # Map timesteps to Karras sigmas directly for multistep sampling
        # See https://github.com/openai/consistency_models/blob/main/cm/karras_diffusion.py#L675
        num_train_timesteps = self.config.num_train_timesteps
        ramp = timesteps[::-1].copy()
        ramp = ramp / (num_train_timesteps - 1)
        sigmas = self._convert_to_karras(ramp)
        timesteps = self.sigma_to_t(sigmas)

        sigmas = np.concatenate([sigmas, [self.sigma_min]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas).to(device=device)

        if str(device).startswith("mps"):
            # mps does not support float64
            self.timesteps = torch.from_numpy(timesteps).to(device, dtype=torch.float32)
        else:
            self.timesteps = torch.from_numpy(timesteps).to(device=device)

    # Modified _convert_to_karras implementation that takes in ramp as argument
    def _convert_to_karras(self, ramp):
        """Constructs the noise schedule of Karras et al. (2022)."""

        sigma_min: float = self.config.sigma_min
        sigma_max: float = self.config.sigma_max

        rho = self.config.rho
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def get_scalings(self, sigma):
        sigma_data = self.config.sigma_data

        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        return c_skip, c_out

    def get_scalings_for_boundary_condition(self, sigma):
        """
        Gets the scalings used in the consistency model parameterization, following Appendix C of the original paper.
        This enforces the consistency model boundary condition.

        Note that `epsilon` in the equations for c_skip and c_out is set to sigma_min.

        Args:
            sigma (`torch.FloatTensor`):
                The current sigma in the Karras sigma schedule.
        Returns:
            `tuple`:
                A two-element tuple where c_skip (which weights the current sample) is the first element and c_out
                (which weights the consistency model output) is the second element.
        """
        sigma_min = self.config.sigma_min
        sigma_data = self.config.sigma_data

        c_skip = sigma_data**2 / ((sigma - sigma_min) ** 2 + sigma_data**2)
        c_out = (sigma - sigma_min) * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        return c_skip, c_out

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[CMStochasticIterativeSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`float`): current timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            generator (`torch.Generator`, *optional*): Random number generator.
            return_dict (`bool`): option for returning tuple rather than EulerDiscreteSchedulerOutput class
        Returns:
            [`~schedulers.scheduling_utils.CMStochasticIterativeSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.CMStochasticIterativeSchedulerOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.
        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    f" `{self.__class__}.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)

        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max

        step_index = self.index_for_timestep(timestep)

        # sigma_next corresponds to next_t in original implementation
        sigma = self.sigmas[step_index]
        if step_index + 1 < self.config.num_train_timesteps:
            sigma_next = self.sigmas[step_index + 1]
        else:
            # Set sigma_next to sigma_min
            sigma_next = self.sigmas[-1]

        # Get scalings for boundary conditions
        c_skip, c_out = self.get_scalings_for_boundary_condition(sigma)

        # 1. Denoise model output using boundary conditions
        denoised = c_out * model_output + c_skip * sample
        if self.config.clip_denoised:
            denoised = denoised.clamp(-1, 1)

        # 2. Sample z ~ N(0, s_noise^2 * I)
        # Noise is not used for onestep sampling.
        if len(self.timesteps) > 1:
            noise = randn_tensor(
                model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
            )
        else:
            noise = torch.zeros_like(model_output)
        z = noise * self.config.s_noise

        sigma_hat = sigma_next.clamp(min=sigma_min, max=sigma_max)

        # 3. Return noisy sample
        # tau = sigma_hat, eps = sigma_min
        prev_sample = denoised + z * (sigma_hat**2 - sigma_min**2) ** 0.5

        if not return_dict:
            return (prev_sample,)

        return CMStochasticIterativeSchedulerOutput(prev_sample=prev_sample)

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
