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

import warnings
from typing import List, Optional, Union

import numpy as np
import PIL
import torch
from PIL import Image

from .configuration_utils import ConfigMixin, register_to_config
from .utils import CONFIG_NAME, PIL_INTERPOLATION, deprecate


class VaeImageProcessor(ConfigMixin):
    """
    Image processor for VAE.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to downscale the image's (height, width) dimensions to multiples of `vae_scale_factor`. Can accept
            `height` and `width` arguments from [`image_processor.VaeImageProcessor.preprocess`] method.
        vae_scale_factor (`int`, *optional*, defaults to `8`):
            VAE scale factor. If `do_resize` is `True`, the image is automatically resized to multiples of this factor.
        resample (`str`, *optional*, defaults to `lanczos`):
            Resampling filter to use when resizing the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image to [-1,1].
        do_convert_rgb (`bool`, *optional*, defaults to be `False`):
            Whether to convert the images to RGB format.
    """

    config_name = CONFIG_NAME

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 8,
        resample: str = "lanczos",
        do_normalize: bool = True,
        do_convert_rgb: bool = False,
    ):
        super().__init__()

    @staticmethod
    def numpy_to_pil(images: np.ndarray) -> PIL.Image.Image:
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @staticmethod
    def pil_to_numpy(images: Union[List[PIL.Image.Image], PIL.Image.Image]) -> np.ndarray:
        """
        Convert a PIL image or a list of PIL images to NumPy arrays.
        """
        if not isinstance(images, list):
            images = [images]
        images = [np.array(image).astype(np.float32) / 255.0 for image in images]
        images = np.stack(images, axis=0)

        return images

    @staticmethod
    def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
        """
        Convert a NumPy image to a PyTorch tensor.
        """
        if images.ndim == 3:
            images = images[..., None]

        images = torch.from_numpy(images.transpose(0, 3, 1, 2))
        return images

    @staticmethod
    def pt_to_numpy(images: torch.FloatTensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a NumPy image.
        """
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        return images

    @staticmethod
    def normalize(images):
        """
        Normalize an image array to [-1,1].
        """
        return 2.0 * images - 1.0

    @staticmethod
    def denormalize(images):
        """
        Denormalize an image array to [0,1].
        """
        return (images / 2 + 0.5).clamp(0, 1)

    @staticmethod
    def convert_to_rgb(image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Converts an image to RGB format.
        """
        image = image.convert("RGB")
        return image

    def resize(
        self,
        image: PIL.Image.Image,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> PIL.Image.Image:
        """
        Resize a PIL image. Both height and width are downscaled to the next integer multiple of `vae_scale_factor`.
        """
        if height is None:
            height = image.height
        if width is None:
            width = image.width

        width, height = (
            x - x % self.config.vae_scale_factor for x in (width, height)
        )  # resize to integer multiple of vae_scale_factor
        image = image.resize((width, height), resample=PIL_INTERPOLATION[self.config.resample])
        return image

    def preprocess(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image, np.ndarray],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Preprocess the image input. Accepted formats are PIL images, NumPy arrays or PyTorch tensors.
        """
        supported_formats = (PIL.Image.Image, np.ndarray, torch.Tensor)
        if isinstance(image, supported_formats):
            image = [image]
        elif not (isinstance(image, list) and all(isinstance(i, supported_formats) for i in image)):
            raise ValueError(
                f"Input is in incorrect format: {[type(i) for i in image]}. Currently, we only support {', '.join(supported_formats)}"
            )

        if isinstance(image[0], PIL.Image.Image):
            if self.config.do_convert_rgb:
                image = [self.convert_to_rgb(i) for i in image]
            if self.config.do_resize:
                image = [self.resize(i, height, width) for i in image]
            image = self.pil_to_numpy(image)  # to np
            image = self.numpy_to_pt(image)  # to pt

        elif isinstance(image[0], np.ndarray):
            image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)
            image = self.numpy_to_pt(image)
            _, _, height, width = image.shape
            if self.config.do_resize and (
                height % self.config.vae_scale_factor != 0 or width % self.config.vae_scale_factor != 0
            ):
                raise ValueError(
                    f"Currently we only support resizing for PIL image - please resize your numpy array to be divisible by {self.config.vae_scale_factor}"
                    f"currently the sizes are {height} and {width}. You can also pass a PIL image instead to use resize option in VAEImageProcessor"
                )

        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, axis=0) if image[0].ndim == 4 else torch.stack(image, axis=0)
            _, channel, height, width = image.shape

            # don't need any preprocess if the image is latents
            if channel == 4:
                return image

            if self.config.do_resize and (
                height % self.config.vae_scale_factor != 0 or width % self.config.vae_scale_factor != 0
            ):
                raise ValueError(
                    f"Currently we only support resizing for PIL image - please resize your pytorch tensor to be divisible by {self.config.vae_scale_factor}"
                    f"currently the sizes are {height} and {width}. You can also pass a PIL image instead to use resize option in VAEImageProcessor"
                )

        # expected range [0,1], normalize to [-1,1]
        do_normalize = self.config.do_normalize
        if image.min() < 0:
            warnings.warn(
                "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
                f"when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
                FutureWarning,
            )
            do_normalize = False

        if do_normalize:
            image = self.normalize(image)

        return image

    def postprocess(
        self,
        image: torch.FloatTensor,
        output_type: str = "pil",
        do_denormalize: Optional[List[bool]] = None,
    ):
        if not isinstance(image, torch.Tensor):
            raise ValueError(
                f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor"
            )
        if output_type not in ["latent", "pt", "np", "pil"]:
            deprecation_message = (
                f"the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of these instead: "
                "`pil`, `np`, `pt`, `latent`"
            )
            deprecate("Unsupported output_type", "1.0.0", deprecation_message, standard_warn=False)
            output_type = "np"

        if output_type == "latent":
            return image

        if do_denormalize is None:
            do_denormalize = [self.config.do_normalize] * image.shape[0]

        image = torch.stack(
            [self.denormalize(image[i]) if do_denormalize[i] else image[i] for i in range(image.shape[0])]
        )

        if output_type == "pt":
            return image

        image = self.pt_to_numpy(image)

        if output_type == "np":
            return image

        if output_type == "pil":
            return self.numpy_to_pil(image)


class VaeImageProcessorLDM3D(VaeImageProcessor):
    """
    Image processor for VAE LDM3D.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to downscale the image's (height, width) dimensions to multiples of `vae_scale_factor`.
        vae_scale_factor (`int`, *optional*, defaults to `8`):
            VAE scale factor. If `do_resize` is `True`, the image is automatically resized to multiples of this factor.
        resample (`str`, *optional*, defaults to `lanczos`):
            Resampling filter to use when resizing the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image to [-1,1].
    """

    config_name = CONFIG_NAME

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 8,
        resample: str = "lanczos",
        do_normalize: bool = True,
    ):
        super().__init__()

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a NumPy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image[:, :, :3]) for image in images]

        return pil_images

    @staticmethod
    def rgblike_to_depthmap(image):
        """
        Args:
            image: RGB-like depth image

        Returns: depth map

        """
        return image[:, :, 1] * 2**8 + image[:, :, 2]

    def numpy_to_depth(self, images):
        """
        Convert a NumPy depth image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images_depth = images[:, :, :, 3:]
        if images.shape[-1] == 6:
            images_depth = (images_depth * 255).round().astype("uint8")
            pil_images = [
                Image.fromarray(self.rgblike_to_depthmap(image_depth), mode="I;16") for image_depth in images_depth
            ]
        elif images.shape[-1] == 4:
            images_depth = (images_depth * 65535.0).astype(np.uint16)
            pil_images = [Image.fromarray(image_depth, mode="I;16") for image_depth in images_depth]
        else:
            raise Exception("Not supported")

        return pil_images

    def postprocess(
        self,
        image: torch.FloatTensor,
        output_type: str = "pil",
        do_denormalize: Optional[List[bool]] = None,
    ):
        if not isinstance(image, torch.Tensor):
            raise ValueError(
                f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor"
            )
        if output_type not in ["latent", "pt", "np", "pil"]:
            deprecation_message = (
                f"the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of these instead: "
                "`pil`, `np`, `pt`, `latent`"
            )
            deprecate("Unsupported output_type", "1.0.0", deprecation_message, standard_warn=False)
            output_type = "np"

        if do_denormalize is None:
            do_denormalize = [self.config.do_normalize] * image.shape[0]

        image = torch.stack(
            [self.denormalize(image[i]) if do_denormalize[i] else image[i] for i in range(image.shape[0])]
        )

        image = self.pt_to_numpy(image)

        if output_type == "np":
            if image.shape[-1] == 6:
                image_depth = np.stack([self.rgblike_to_depthmap(im[:, :, 3:]) for im in image], axis=0)
            else:
                image_depth = image[:, :, :, 3:]
            return image[:, :, :, :3], image_depth

        if output_type == "pil":
            return self.numpy_to_pil(image), self.numpy_to_depth(image)
        else:
            raise Exception(f"This type {output_type} is not supported")
