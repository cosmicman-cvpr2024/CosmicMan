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
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn

from .models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    AttnProcessor,
    AttnProcessor2_0,
    CustomDiffusionAttnProcessor,
    CustomDiffusionXFormersAttnProcessor,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    LoRALinearLayer,
    LoRAXFormersAttnProcessor,
    SlicedAttnAddedKVProcessor,
    XFormersAttnProcessor,
)
from .utils import (
    DIFFUSERS_CACHE,
    HF_HUB_OFFLINE,
    _get_model_file,
    deprecate,
    is_safetensors_available,
    is_transformers_available,
    logging,
)


if is_safetensors_available():
    import safetensors

if is_transformers_available():
    from transformers import CLIPTextModel, PreTrainedModel, PreTrainedTokenizer


logger = logging.get_logger(__name__)

TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"

LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"

TEXT_INVERSION_NAME = "learned_embeds.bin"
TEXT_INVERSION_NAME_SAFE = "learned_embeds.safetensors"

CUSTOM_DIFFUSION_WEIGHT_NAME = "pytorch_custom_diffusion_weights.bin"
CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE = "pytorch_custom_diffusion_weights.safetensors"


class PatchedLoraProjection(nn.Module):
    def __init__(self, regular_linear_layer, lora_scale=1, network_alpha=None, rank=4, dtype=None):
        super().__init__()
        self.regular_linear_layer = regular_linear_layer

        device = self.regular_linear_layer.weight.device

        if dtype is None:
            dtype = self.regular_linear_layer.weight.dtype

        self.lora_linear_layer = LoRALinearLayer(
            self.regular_linear_layer.in_features,
            self.regular_linear_layer.out_features,
            network_alpha=network_alpha,
            device=device,
            dtype=dtype,
            rank=rank,
        )

        self.lora_scale = lora_scale

    def forward(self, input):
        return self.regular_linear_layer(input) + self.lora_scale * self.lora_linear_layer(input)


def text_encoder_attn_modules(text_encoder):
    attn_modules = []

    if isinstance(text_encoder, CLIPTextModel):
        for i, layer in enumerate(text_encoder.text_model.encoder.layers):
            name = f"text_model.encoder.layers.{i}.self_attn"
            mod = layer.self_attn
            attn_modules.append((name, mod))
    else:
        raise ValueError(f"do not know how to get attention modules for: {text_encoder.__class__.__name__}")

    return attn_modules


def text_encoder_lora_state_dict(text_encoder):
    state_dict = {}

    for name, module in text_encoder_attn_modules(text_encoder):
        for k, v in module.q_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.q_proj.lora_linear_layer.{k}"] = v

        for k, v in module.k_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.k_proj.lora_linear_layer.{k}"] = v

        for k, v in module.v_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.v_proj.lora_linear_layer.{k}"] = v

        for k, v in module.out_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.out_proj.lora_linear_layer.{k}"] = v

    return state_dict


class AttnProcsLayers(torch.nn.Module):
    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        super().__init__()
        self.layers = torch.nn.ModuleList(state_dict.values())
        self.mapping = dict(enumerate(state_dict.keys()))
        self.rev_mapping = {v: k for k, v in enumerate(state_dict.keys())}

        # .processor for unet, .self_attn for text encoder
        self.split_keys = [".processor", ".self_attn"]

        # we add a hook to state_dict() and load_state_dict() so that the
        # naming fits with `unet.attn_processors`
        def map_to(module, state_dict, *args, **kwargs):
            new_state_dict = {}
            for key, value in state_dict.items():
                num = int(key.split(".")[1])  # 0 is always "layers"
                new_key = key.replace(f"layers.{num}", module.mapping[num])
                new_state_dict[new_key] = value

            return new_state_dict

        def remap_key(key, state_dict):
            for k in self.split_keys:
                if k in key:
                    return key.split(k)[0] + k

            raise ValueError(
                f"There seems to be a problem with the state_dict: {set(state_dict.keys())}. {key} has to have one of {self.split_keys}."
            )

        def map_from(module, state_dict, *args, **kwargs):
            all_keys = list(state_dict.keys())
            for key in all_keys:
                replace_key = remap_key(key, state_dict)
                new_key = key.replace(replace_key, f"layers.{module.rev_mapping[replace_key]}")
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        self._register_state_dict_hook(map_to)
        self._register_load_state_dict_pre_hook(map_from, with_module=True)


class UNet2DConditionLoadersMixin:
    text_encoder_name = TEXT_ENCODER_NAME
    unet_name = UNET_NAME

    def load_attn_procs(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], **kwargs):
        r"""
        Load pretrained attention processor layers into [`UNet2DConditionModel`]. Attention processor layers have to be
        defined in
        [`cross_attention.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py)
        and be a `torch.nn.Module` class.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the model id (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a directory (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.

        """

        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        network_alpha = kwargs.pop("network_alpha", None)

        if use_safetensors and not is_safetensors_available():
            raise ValueError(
                "`use_safetensors`=True but safetensors is not installed. Please install safetensors with `pip install safetensors"
            )

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = is_safetensors_available()
            allow_pickle = True

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        model_file = None
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            # Let's first try to load .safetensors weights
            if (use_safetensors and weight_name is None) or (
                weight_name is not None and weight_name.endswith(".safetensors")
            ):
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=weight_name or LORA_WEIGHT_NAME_SAFE,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                    )
                    state_dict = safetensors.torch.load_file(model_file, device="cpu")
                except IOError as e:
                    if not allow_pickle:
                        raise e
                    # try loading non-safetensors weights
                    pass
            if model_file is None:
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name or LORA_WEIGHT_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        # fill attn processors
        attn_processors = {}

        is_lora = all("lora" in k for k in state_dict.keys())
        is_custom_diffusion = any("custom_diffusion" in k for k in state_dict.keys())

        if is_lora:
            is_new_lora_format = all(
                key.startswith(self.unet_name) or key.startswith(self.text_encoder_name) for key in state_dict.keys()
            )
            if is_new_lora_format:
                # Strip the `"unet"` prefix.
                is_text_encoder_present = any(key.startswith(self.text_encoder_name) for key in state_dict.keys())
                if is_text_encoder_present:
                    warn_message = "The state_dict contains LoRA params corresponding to the text encoder which are not being used here. To use both UNet and text encoder related LoRA params, use [`pipe.load_lora_weights()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraLoaderMixin.load_lora_weights)."
                    warnings.warn(warn_message)
                unet_keys = [k for k in state_dict.keys() if k.startswith(self.unet_name)]
                state_dict = {k.replace(f"{self.unet_name}.", ""): v for k, v in state_dict.items() if k in unet_keys}

            lora_grouped_dict = defaultdict(dict)
            for key, value in state_dict.items():
                attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                lora_grouped_dict[attn_processor_key][sub_key] = value

            for key, value_dict in lora_grouped_dict.items():
                rank = value_dict["to_k_lora.down.weight"].shape[0]
                hidden_size = value_dict["to_k_lora.up.weight"].shape[0]

                attn_processor = self
                for sub_key in key.split("."):
                    attn_processor = getattr(attn_processor, sub_key)

                if isinstance(
                    attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)
                ):
                    cross_attention_dim = value_dict["add_k_proj_lora.down.weight"].shape[1]
                    attn_processor_class = LoRAAttnAddedKVProcessor
                else:
                    cross_attention_dim = value_dict["to_k_lora.down.weight"].shape[1]
                    if isinstance(attn_processor, (XFormersAttnProcessor, LoRAXFormersAttnProcessor)):
                        attn_processor_class = LoRAXFormersAttnProcessor
                    else:
                        attn_processor_class = (
                            LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
                        )

                attn_processors[key] = attn_processor_class(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=rank,
                    network_alpha=network_alpha,
                )
                attn_processors[key].load_state_dict(value_dict)
        elif is_custom_diffusion:
            custom_diffusion_grouped_dict = defaultdict(dict)
            for key, value in state_dict.items():
                if len(value) == 0:
                    custom_diffusion_grouped_dict[key] = {}
                else:
                    if "to_out" in key:
                        attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                    else:
                        attn_processor_key, sub_key = ".".join(key.split(".")[:-2]), ".".join(key.split(".")[-2:])
                    custom_diffusion_grouped_dict[attn_processor_key][sub_key] = value

            for key, value_dict in custom_diffusion_grouped_dict.items():
                if len(value_dict) == 0:
                    attn_processors[key] = CustomDiffusionAttnProcessor(
                        train_kv=False, train_q_out=False, hidden_size=None, cross_attention_dim=None
                    )
                else:
                    cross_attention_dim = value_dict["to_k_custom_diffusion.weight"].shape[1]
                    hidden_size = value_dict["to_k_custom_diffusion.weight"].shape[0]
                    train_q_out = True if "to_q_custom_diffusion.weight" in value_dict else False
                    attn_processors[key] = CustomDiffusionAttnProcessor(
                        train_kv=True,
                        train_q_out=train_q_out,
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                    )
                    attn_processors[key].load_state_dict(value_dict)
        else:
            raise ValueError(
                f"{model_file} does not seem to be in the correct format expected by LoRA or Custom Diffusion training."
            )

        # set correct dtype & device
        attn_processors = {k: v.to(device=self.device, dtype=self.dtype) for k, v in attn_processors.items()}

        # set layers
        self.set_attn_processor(attn_processors)

    def save_attn_procs(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = False,
        **kwargs,
    ):
        r"""
        Save an attention processor to a directory so that it can be reloaded using the
        [`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`] method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save an attention processor to. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.

        """
        weight_name = weight_name or deprecate(
            "weights_name",
            "0.20.0",
            "`weights_name` is deprecated, please use `weight_name` instead.",
            take_from=kwargs,
        )
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if save_function is None:
            if safe_serialization:

                def save_function(weights, filename):
                    return safetensors.torch.save_file(weights, filename, metadata={"format": "pt"})

            else:
                save_function = torch.save

        os.makedirs(save_directory, exist_ok=True)

        is_custom_diffusion = any(
            isinstance(x, (CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor))
            for (_, x) in self.attn_processors.items()
        )
        if is_custom_diffusion:
            model_to_save = AttnProcsLayers(
                {
                    y: x
                    for (y, x) in self.attn_processors.items()
                    if isinstance(x, (CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor))
                }
            )
            state_dict = model_to_save.state_dict()
            for name, attn in self.attn_processors.items():
                if len(attn.state_dict()) == 0:
                    state_dict[name] = {}
        else:
            model_to_save = AttnProcsLayers(self.attn_processors)
            state_dict = model_to_save.state_dict()

        if weight_name is None:
            if safe_serialization:
                weight_name = CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE if is_custom_diffusion else LORA_WEIGHT_NAME_SAFE
            else:
                weight_name = CUSTOM_DIFFUSION_WEIGHT_NAME if is_custom_diffusion else LORA_WEIGHT_NAME

        # Save the model
        save_function(state_dict, os.path.join(save_directory, weight_name))
        logger.info(f"Model weights saved in {os.path.join(save_directory, weight_name)}")


class TextualInversionLoaderMixin:
    r"""
    Load textual inversion tokens and embeddings to the tokenizer and text encoder.
    """

    def maybe_convert_prompt(self, prompt: Union[str, List[str]], tokenizer: "PreTrainedTokenizer"):
        r"""
        Processes prompts that include a special token corresponding to a multi-vector textual inversion embedding to
        be replaced with multiple special tokens each corresponding to one of the vectors. If the prompt has no textual
        inversion token or if the textual inversion token is a single vector, the input prompt is returned.

        Parameters:
            prompt (`str` or list of `str`):
                The prompt or prompts to guide the image generation.
            tokenizer (`PreTrainedTokenizer`):
                The tokenizer responsible for encoding the prompt into input tokens.

        Returns:
            `str` or list of `str`: The converted prompt
        """
        if not isinstance(prompt, List):
            prompts = [prompt]
        else:
            prompts = prompt

        prompts = [self._maybe_convert_prompt(p, tokenizer) for p in prompts]

        if not isinstance(prompt, List):
            return prompts[0]

        return prompts

    def _maybe_convert_prompt(self, prompt: str, tokenizer: "PreTrainedTokenizer"):
        r"""
        Maybe convert a prompt into a "multi vector"-compatible prompt. If the prompt includes a token that corresponds
        to a multi-vector textual inversion embedding, this function will process the prompt so that the special token
        is replaced with multiple special tokens each corresponding to one of the vectors. If the prompt has no textual
        inversion token or a textual inversion token that is a single vector, the input prompt is simply returned.

        Parameters:
            prompt (`str`):
                The prompt to guide the image generation.
            tokenizer (`PreTrainedTokenizer`):
                The tokenizer responsible for encoding the prompt into input tokens.

        Returns:
            `str`: The converted prompt
        """
        tokens = tokenizer.tokenize(prompt)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token in tokenizer.added_tokens_encoder:
                replacement = token
                i = 1
                while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                    replacement += f" {token}_{i}"
                    i += 1

                prompt = prompt.replace(token, replacement)

        return prompt

    def load_textual_inversion(
        self,
        pretrained_model_name_or_path: Union[str, List[str], Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
        token: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        r"""
        Load textual inversion embeddings into the text encoder of [`StableDiffusionPipeline`] (both 🤗 Diffusers and
        Automatic1111 formats are supported).

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike` or `List[str or os.PathLike]` or `Dict` or `List[Dict]`):
                Can be either one of the following or a list of them:

                    - A string, the *model id* (for example `sd-concepts-library/low-poly-hd-logos-icons`) of a
                      pretrained model hosted on the Hub.
                    - A path to a *directory* (for example `./my_text_inversion_directory/`) containing the textual
                      inversion weights.
                    - A path to a *file* (for example `./my_text_inversions.pt`) containing textual inversion weights.
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            token (`str` or `List[str]`, *optional*):
                Override the token to use for the textual inversion weights. If `pretrained_model_name_or_path` is a
                list, then `token` must also be a list of equal length.
            weight_name (`str`, *optional*):
                Name of a custom weight file. This should be used when:

                    - The saved textual inversion file is in 🤗 Diffusers format, but was saved under a specific weight
                      name such as `text_inv.bin`.
                    - The saved textual inversion file is in the Automatic1111 format.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.

        Example:

        To load a textual inversion embedding vector in 🤗 Diffusers format:

        ```py
        from diffusers import StableDiffusionPipeline
        import torch

        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

        pipe.load_textual_inversion("sd-concepts-library/cat-toy")

        prompt = "A <cat-toy> backpack"

        image = pipe(prompt, num_inference_steps=50).images[0]
        image.save("cat-backpack.png")
        ```

        To load a textual inversion embedding vector in Automatic1111 format, make sure to download the vector first
        (for example from [civitAI](https://civitai.com/models/3036?modelVersionId=9857)) and then load the vector
        locally:

        ```py
        from diffusers import StableDiffusionPipeline
        import torch

        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

        pipe.load_textual_inversion("./charturnerv2.pt", token="charturnerv2")

        prompt = "charturnerv2, multiple views of the same character in the same outfit, a character turnaround of a woman wearing a black jacket and red shirt, best quality, intricate details."

        image = pipe(prompt, num_inference_steps=50).images[0]
        image.save("character.png")
        ```

        """
        if not hasattr(self, "tokenizer") or not isinstance(self.tokenizer, PreTrainedTokenizer):
            raise ValueError(
                f"{self.__class__.__name__} requires `self.tokenizer` of type `PreTrainedTokenizer` for calling"
                f" `{self.load_textual_inversion.__name__}`"
            )

        if not hasattr(self, "text_encoder") or not isinstance(self.text_encoder, PreTrainedModel):
            raise ValueError(
                f"{self.__class__.__name__} requires `self.text_encoder` of type `PreTrainedModel` for calling"
                f" `{self.load_textual_inversion.__name__}`"
            )

        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        if use_safetensors and not is_safetensors_available():
            raise ValueError(
                "`use_safetensors`=True but safetensors is not installed. Please install safetensors with `pip install safetensors"
            )

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = is_safetensors_available()
            allow_pickle = True

        user_agent = {
            "file_type": "text_inversion",
            "framework": "pytorch",
        }

        if not isinstance(pretrained_model_name_or_path, list):
            pretrained_model_name_or_paths = [pretrained_model_name_or_path]
        else:
            pretrained_model_name_or_paths = pretrained_model_name_or_path

        if isinstance(token, str):
            tokens = [token]
        elif token is None:
            tokens = [None] * len(pretrained_model_name_or_paths)
        else:
            tokens = token

        if len(pretrained_model_name_or_paths) != len(tokens):
            raise ValueError(
                f"You have passed a list of models of length {len(pretrained_model_name_or_paths)}, and list of tokens of length {len(tokens)}"
                f"Make sure both lists have the same length."
            )

        valid_tokens = [t for t in tokens if t is not None]
        if len(set(valid_tokens)) < len(valid_tokens):
            raise ValueError(f"You have passed a list of tokens that contains duplicates: {tokens}")

        token_ids_and_embeddings = []

        for pretrained_model_name_or_path, token in zip(pretrained_model_name_or_paths, tokens):
            if not isinstance(pretrained_model_name_or_path, dict):
                # 1. Load textual inversion file
                model_file = None
                # Let's first try to load .safetensors weights
                if (use_safetensors and weight_name is None) or (
                    weight_name is not None and weight_name.endswith(".safetensors")
                ):
                    try:
                        model_file = _get_model_file(
                            pretrained_model_name_or_path,
                            weights_name=weight_name or TEXT_INVERSION_NAME_SAFE,
                            cache_dir=cache_dir,
                            force_download=force_download,
                            resume_download=resume_download,
                            proxies=proxies,
                            local_files_only=local_files_only,
                            use_auth_token=use_auth_token,
                            revision=revision,
                            subfolder=subfolder,
                            user_agent=user_agent,
                        )
                        state_dict = safetensors.torch.load_file(model_file, device="cpu")
                    except Exception as e:
                        if not allow_pickle:
                            raise e

                        model_file = None

                if model_file is None:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=weight_name or TEXT_INVERSION_NAME,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                    )
                    state_dict = torch.load(model_file, map_location="cpu")
            else:
                state_dict = pretrained_model_name_or_path

            # 2. Load token and embedding correcly from file
            loaded_token = None
            if isinstance(state_dict, torch.Tensor):
                if token is None:
                    raise ValueError(
                        "You are trying to load a textual inversion embedding that has been saved as a PyTorch tensor. Make sure to pass the name of the corresponding token in this case: `token=...`."
                    )
                embedding = state_dict
            elif len(state_dict) == 1:
                # diffusers
                loaded_token, embedding = next(iter(state_dict.items()))
            elif "string_to_param" in state_dict:
                # A1111
                loaded_token = state_dict["name"]
                embedding = state_dict["string_to_param"]["*"]

            if token is not None and loaded_token != token:
                logger.info(f"The loaded token: {loaded_token} is overwritten by the passed token {token}.")
            else:
                token = loaded_token

            embedding = embedding.to(dtype=self.text_encoder.dtype, device=self.text_encoder.device)

            # 3. Make sure we don't mess up the tokenizer or text encoder
            vocab = self.tokenizer.get_vocab()
            if token in vocab:
                raise ValueError(
                    f"Token {token} already in tokenizer vocabulary. Please choose a different token name or remove {token} and embedding from the tokenizer and text encoder."
                )
            elif f"{token}_1" in vocab:
                multi_vector_tokens = [token]
                i = 1
                while f"{token}_{i}" in self.tokenizer.added_tokens_encoder:
                    multi_vector_tokens.append(f"{token}_{i}")
                    i += 1

                raise ValueError(
                    f"Multi-vector Token {multi_vector_tokens} already in tokenizer vocabulary. Please choose a different token name or remove the {multi_vector_tokens} and embedding from the tokenizer and text encoder."
                )

            is_multi_vector = len(embedding.shape) > 1 and embedding.shape[0] > 1

            if is_multi_vector:
                tokens = [token] + [f"{token}_{i}" for i in range(1, embedding.shape[0])]
                embeddings = [e for e in embedding]  # noqa: C416
            else:
                tokens = [token]
                embeddings = [embedding[0]] if len(embedding.shape) > 1 else [embedding]

            # add tokens and get ids
            self.tokenizer.add_tokens(tokens)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            token_ids_and_embeddings += zip(token_ids, embeddings)

            logger.info(f"Loaded textual inversion embedding for {token}.")

        # resize token embeddings and set all new embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        for token_id, embedding in token_ids_and_embeddings:
            self.text_encoder.get_input_embeddings().weight.data[token_id] = embedding


class LoraLoaderMixin:
    r"""
    Load LoRA layers into [`UNet2DConditionModel`] and
    [`CLIPTextModel`](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel).
    """
    text_encoder_name = TEXT_ENCODER_NAME
    unet_name = UNET_NAME

    def load_lora_weights(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], **kwargs):
        """
        Load LoRA weights specified in `pretrained_model_name_or_path_or_dict` into self.unet and self.text_encoder.

        All kwargs are forwarded to `self.lora_state_dict`.

        See [`~loaders.LoraLoaderMixin.lora_state_dict`] for more details on how the state dict is loaded.

        See [`~loaders.LoraLoaderMixin.load_lora_into_unet`] for more details on how the state dict is loaded into
        `self.unet`.

        See [`~loaders.LoraLoaderMixin.load_lora_into_text_encoder`] for more details on how the state dict is loaded
        into `self.text_encoder`.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                See [`~loaders.LoraLoaderMixin.lora_state_dict`].

            kwargs:
                See [`~loaders.LoraLoaderMixin.lora_state_dict`].
        """
        state_dict, network_alpha = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)
        self.load_lora_into_unet(state_dict, network_alpha=network_alpha, unet=self.unet)
        self.load_lora_into_text_encoder(
            state_dict, network_alpha=network_alpha, text_encoder=self.text_encoder, lora_scale=self.lora_scale
        )

    @classmethod
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        **kwargs,
    ):
        r"""
        Return state dict for lora weights

        <Tip warning={true}>

        We support loading A1111 formatted LoRA checkpoints in a limited capacity.

        This function is experimental and might change in the future.

        </Tip>

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.

        """
        # Load the main state dict first which has the LoRA layers for either of
        # UNet and text encoder or both.
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        if use_safetensors and not is_safetensors_available():
            raise ValueError(
                "`use_safetensors`=True but safetensors is not installed. Please install safetensors with `pip install safetensors"
            )

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = is_safetensors_available()
            allow_pickle = True

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        model_file = None
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            # Let's first try to load .safetensors weights
            if (use_safetensors and weight_name is None) or (
                weight_name is not None and weight_name.endswith(".safetensors")
            ):
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=weight_name or LORA_WEIGHT_NAME_SAFE,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                    )
                    state_dict = safetensors.torch.load_file(model_file, device="cpu")
                except (IOError, safetensors.SafetensorError) as e:
                    if not allow_pickle:
                        raise e
                    # try loading non-safetensors weights
                    pass
            if model_file is None:
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name or LORA_WEIGHT_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        # Convert kohya-ss Style LoRA attn procs to diffusers attn procs
        network_alpha = None
        if all((k.startswith("lora_te_") or k.startswith("lora_unet_")) for k in state_dict.keys()):
            state_dict, network_alpha = cls._convert_kohya_lora_to_diffusers(state_dict)

        return state_dict, network_alpha

    @classmethod
    def load_lora_into_unet(cls, state_dict, network_alpha, unet):
        """
        This will load the LoRA layers specified in `state_dict` into `unet`

        Parameters:
            state_dict (`dict`):
                A standard state dict containing the lora layer parameters. The keys can either be indexed directly
                into the unet or prefixed with an additional `unet` which can be used to distinguish between text
                encoder lora layers.
            network_alpha (`float`):
                See `LoRALinearLayer` for more details.
            unet (`UNet2DConditionModel`):
                The UNet model to load the LoRA layers into.
        """

        # If the serialization format is new (introduced in https://github.com/huggingface/diffusers/pull/2918),
        # then the `state_dict` keys should have `self.unet_name` and/or `self.text_encoder_name` as
        # their prefixes.
        keys = list(state_dict.keys())
        if all(key.startswith(cls.unet_name) or key.startswith(cls.text_encoder_name) for key in keys):
            # Load the layers corresponding to UNet.
            unet_keys = [k for k in keys if k.startswith(cls.unet_name)]
            logger.info(f"Loading {cls.unet_name}.")
            unet_lora_state_dict = {
                k.replace(f"{cls.unet_name}.", ""): v for k, v in state_dict.items() if k in unet_keys
            }
            unet.load_attn_procs(unet_lora_state_dict, network_alpha=network_alpha)

        # Otherwise, we're dealing with the old format. This means the `state_dict` should only
        # contain the module names of the `unet` as its keys WITHOUT any prefix.
        elif not all(
            key.startswith(cls.unet_name) or key.startswith(cls.text_encoder_name) for key in state_dict.keys()
        ):
            unet.load_attn_procs(state_dict)
            warn_message = "You have saved the LoRA weights using the old format. To convert the old LoRA weights to the new format, you can first load them in a dictionary and then create a new dictionary like the following: `new_state_dict = {f'unet'.{module_name}: params for module_name, params in old_state_dict.items()}`."
            warnings.warn(warn_message)

    @classmethod
    def load_lora_into_text_encoder(cls, state_dict, network_alpha, text_encoder, lora_scale=1.0):
        """
        This will load the LoRA layers specified in `state_dict` into `text_encoder`

        Parameters:
            state_dict (`dict`):
                A standard state dict containing the lora layer parameters. The key shoult be prefixed with an
                additional `text_encoder` to distinguish between unet lora layers.
            network_alpha (`float`):
                See `LoRALinearLayer` for more details.
            text_encoder (`CLIPTextModel`):
                The text encoder model to load the LoRA layers into.
            lora_scale (`float`):
                How much to scale the output of the lora linear layer before it is added with the output of the regular
                lora layer.
        """

        # If the serialization format is new (introduced in https://github.com/huggingface/diffusers/pull/2918),
        # then the `state_dict` keys should have `self.unet_name` and/or `self.text_encoder_name` as
        # their prefixes.
        keys = list(state_dict.keys())
        if all(key.startswith(cls.unet_name) or key.startswith(cls.text_encoder_name) for key in keys):
            # Load the layers corresponding to text encoder and make necessary adjustments.
            text_encoder_keys = [k for k in keys if k.startswith(cls.text_encoder_name)]
            text_encoder_lora_state_dict = {
                k.replace(f"{cls.text_encoder_name}.", ""): v for k, v in state_dict.items() if k in text_encoder_keys
            }
            if len(text_encoder_lora_state_dict) > 0:
                logger.info(f"Loading {cls.text_encoder_name}.")

                if any("to_out_lora" in k for k in text_encoder_lora_state_dict.keys()):
                    # Convert from the old naming convention to the new naming convention.
                    #
                    # Previously, the old LoRA layers were stored on the state dict at the
                    # same level as the attention block i.e.
                    # `text_model.encoder.layers.11.self_attn.to_out_lora.up.weight`.
                    #
                    # This is no actual module at that point, they were monkey patched on to the
                    # existing module. We want to be able to load them via their actual state dict.
                    # They're in `PatchedLoraProjection.lora_linear_layer` now.
                    for name, _ in text_encoder_attn_modules(text_encoder):
                        text_encoder_lora_state_dict[
                            f"{name}.q_proj.lora_linear_layer.up.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_q_lora.up.weight")
                        text_encoder_lora_state_dict[
                            f"{name}.k_proj.lora_linear_layer.up.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_k_lora.up.weight")
                        text_encoder_lora_state_dict[
                            f"{name}.v_proj.lora_linear_layer.up.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_v_lora.up.weight")
                        text_encoder_lora_state_dict[
                            f"{name}.out_proj.lora_linear_layer.up.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_out_lora.up.weight")

                        text_encoder_lora_state_dict[
                            f"{name}.q_proj.lora_linear_layer.down.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_q_lora.down.weight")
                        text_encoder_lora_state_dict[
                            f"{name}.k_proj.lora_linear_layer.down.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_k_lora.down.weight")
                        text_encoder_lora_state_dict[
                            f"{name}.v_proj.lora_linear_layer.down.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_v_lora.down.weight")
                        text_encoder_lora_state_dict[
                            f"{name}.out_proj.lora_linear_layer.down.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_out_lora.down.weight")

                rank = text_encoder_lora_state_dict[
                    "text_model.encoder.layers.0.self_attn.out_proj.lora_linear_layer.up.weight"
                ].shape[1]

                cls._modify_text_encoder(text_encoder, lora_scale, network_alpha, rank=rank)

                # set correct dtype & device
                text_encoder_lora_state_dict = {
                    k: v.to(device=text_encoder.device, dtype=text_encoder.dtype)
                    for k, v in text_encoder_lora_state_dict.items()
                }

                load_state_dict_results = text_encoder.load_state_dict(text_encoder_lora_state_dict, strict=False)
                if len(load_state_dict_results.unexpected_keys) != 0:
                    raise ValueError(
                        f"failed to load text encoder state dict, unexpected keys: {load_state_dict_results.unexpected_keys}"
                    )

    @property
    def lora_scale(self) -> float:
        # property function that returns the lora scale which can be set at run time by the pipeline.
        # if _lora_scale has not been set, return 1
        return self._lora_scale if hasattr(self, "_lora_scale") else 1.0

    def _remove_text_encoder_monkey_patch(self):
        self._remove_text_encoder_monkey_patch_classmethod(self.text_encoder)

    @classmethod
    def _remove_text_encoder_monkey_patch_classmethod(cls, text_encoder):
        for _, attn_module in text_encoder_attn_modules(text_encoder):
            if isinstance(attn_module.q_proj, PatchedLoraProjection):
                attn_module.q_proj = attn_module.q_proj.regular_linear_layer
                attn_module.k_proj = attn_module.k_proj.regular_linear_layer
                attn_module.v_proj = attn_module.v_proj.regular_linear_layer
                attn_module.out_proj = attn_module.out_proj.regular_linear_layer

    @classmethod
    def _modify_text_encoder(cls, text_encoder, lora_scale=1, network_alpha=None, rank=4, dtype=None):
        r"""
        Monkey-patches the forward passes of attention modules of the text encoder.
        """

        # First, remove any monkey-patch that might have been applied before
        cls._remove_text_encoder_monkey_patch_classmethod(text_encoder)

        lora_parameters = []

        for _, attn_module in text_encoder_attn_modules(text_encoder):
            attn_module.q_proj = PatchedLoraProjection(
                attn_module.q_proj, lora_scale, network_alpha, rank=rank, dtype=dtype
            )
            lora_parameters.extend(attn_module.q_proj.lora_linear_layer.parameters())

            attn_module.k_proj = PatchedLoraProjection(
                attn_module.k_proj, lora_scale, network_alpha, rank=rank, dtype=dtype
            )
            lora_parameters.extend(attn_module.k_proj.lora_linear_layer.parameters())

            attn_module.v_proj = PatchedLoraProjection(
                attn_module.v_proj, lora_scale, network_alpha, rank=rank, dtype=dtype
            )
            lora_parameters.extend(attn_module.v_proj.lora_linear_layer.parameters())

            attn_module.out_proj = PatchedLoraProjection(
                attn_module.out_proj, lora_scale, network_alpha, rank=rank, dtype=dtype
            )
            lora_parameters.extend(attn_module.out_proj.lora_linear_layer.parameters())

        return lora_parameters

    @classmethod
    def save_lora_weights(
        self,
        save_directory: Union[str, os.PathLike],
        unet_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        text_encoder_lora_layers: Dict[str, torch.nn.Module] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = False,
    ):
        r"""
        Save the LoRA parameters corresponding to the UNet and text encoder.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save LoRA parameters to. Will be created if it doesn't exist.
            unet_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the UNet.
            text_encoder_lora_layers (`Dict[str, torch.nn.Module] or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `text_encoder`. Must explicitly pass the text
                encoder LoRA state dict because it comes 🤗 Transformers.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if save_function is None:
            if safe_serialization:

                def save_function(weights, filename):
                    return safetensors.torch.save_file(weights, filename, metadata={"format": "pt"})

            else:
                save_function = torch.save

        os.makedirs(save_directory, exist_ok=True)

        # Create a flat dictionary.
        state_dict = {}
        if unet_lora_layers is not None:
            weights = (
                unet_lora_layers.state_dict() if isinstance(unet_lora_layers, torch.nn.Module) else unet_lora_layers
            )

            unet_lora_state_dict = {f"{self.unet_name}.{module_name}": param for module_name, param in weights.items()}
            state_dict.update(unet_lora_state_dict)

        if text_encoder_lora_layers is not None:
            weights = (
                text_encoder_lora_layers.state_dict()
                if isinstance(text_encoder_lora_layers, torch.nn.Module)
                else text_encoder_lora_layers
            )

            text_encoder_lora_state_dict = {
                f"{self.text_encoder_name}.{module_name}": param for module_name, param in weights.items()
            }
            state_dict.update(text_encoder_lora_state_dict)

        # Save the model
        if weight_name is None:
            if safe_serialization:
                weight_name = LORA_WEIGHT_NAME_SAFE
            else:
                weight_name = LORA_WEIGHT_NAME

        save_function(state_dict, os.path.join(save_directory, weight_name))
        logger.info(f"Model weights saved in {os.path.join(save_directory, weight_name)}")

    @classmethod
    def _convert_kohya_lora_to_diffusers(cls, state_dict):
        unet_state_dict = {}
        te_state_dict = {}
        network_alpha = None

        for key, value in state_dict.items():
            if "lora_down" in key:
                lora_name = key.split(".")[0]
                lora_name_up = lora_name + ".lora_up.weight"
                lora_name_alpha = lora_name + ".alpha"
                if lora_name_alpha in state_dict:
                    alpha = state_dict[lora_name_alpha].item()
                    if network_alpha is None:
                        network_alpha = alpha
                    elif network_alpha != alpha:
                        raise ValueError("Network alpha is not consistent")

                if lora_name.startswith("lora_unet_"):
                    diffusers_name = key.replace("lora_unet_", "").replace("_", ".")
                    diffusers_name = diffusers_name.replace("down.blocks", "down_blocks")
                    diffusers_name = diffusers_name.replace("mid.block", "mid_block")
                    diffusers_name = diffusers_name.replace("up.blocks", "up_blocks")
                    diffusers_name = diffusers_name.replace("transformer.blocks", "transformer_blocks")
                    diffusers_name = diffusers_name.replace("to.q.lora", "to_q_lora")
                    diffusers_name = diffusers_name.replace("to.k.lora", "to_k_lora")
                    diffusers_name = diffusers_name.replace("to.v.lora", "to_v_lora")
                    diffusers_name = diffusers_name.replace("to.out.0.lora", "to_out_lora")
                    if "transformer_blocks" in diffusers_name:
                        if "attn1" in diffusers_name or "attn2" in diffusers_name:
                            diffusers_name = diffusers_name.replace("attn1", "attn1.processor")
                            diffusers_name = diffusers_name.replace("attn2", "attn2.processor")
                            unet_state_dict[diffusers_name] = value
                            unet_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict[lora_name_up]
                elif lora_name.startswith("lora_te_"):
                    diffusers_name = key.replace("lora_te_", "").replace("_", ".")
                    diffusers_name = diffusers_name.replace("text.model", "text_model")
                    diffusers_name = diffusers_name.replace("self.attn", "self_attn")
                    diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
                    diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
                    diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
                    diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
                    if "self_attn" in diffusers_name:
                        te_state_dict[diffusers_name] = value
                        te_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict[lora_name_up]

        unet_state_dict = {f"{UNET_NAME}.{module_name}": params for module_name, params in unet_state_dict.items()}
        te_state_dict = {f"{TEXT_ENCODER_NAME}.{module_name}": params for module_name, params in te_state_dict.items()}
        new_state_dict = {**unet_state_dict, **te_state_dict}
        return new_state_dict, network_alpha

    def unload_lora_weights(self):
        """
        Unloads the LoRA parameters.

        Examples:

        ```python
        >>> # Assuming `pipeline` is already loaded with the LoRA parameters.
        >>> pipeline.unload_lora_weights()
        >>> ...
        ```
        """
        is_unet_lora = all(
            isinstance(processor, (LoRAAttnProcessor2_0, LoRAAttnProcessor, LoRAAttnAddedKVProcessor))
            for _, processor in self.unet.attn_processors.items()
        )
        # Handle attention processors that are a mix of regular attention and AddedKV
        # attention.
        if is_unet_lora:
            is_attn_procs_mixed = all(
                isinstance(processor, (LoRAAttnProcessor2_0, LoRAAttnProcessor))
                for _, processor in self.unet.attn_processors.items()
            )
            if not is_attn_procs_mixed:
                unet_attn_proc_cls = AttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else AttnProcessor
                self.unet.set_attn_processor(unet_attn_proc_cls())
            else:
                self.unet.set_default_attn_processor()

        # Safe to call the following regardless of LoRA.
        self._remove_text_encoder_monkey_patch()


class FromSingleFileMixin:
    """
    Load model weights saved in the `.ckpt` format into a [`DiffusionPipeline`].
    """

    @classmethod
    def from_ckpt(cls, *args, **kwargs):
        deprecation_message = "The function `from_ckpt` is deprecated in favor of `from_single_file` and will be removed in diffusers v.0.21. Please make sure to use `StableDiffusionPipeline.from_single_file(...)` instead."
        deprecate("from_ckpt", "0.21.0", deprecation_message, standard_warn=False)
        return cls.from_single_file(*args, **kwargs)

    @classmethod
    def from_single_file(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Instantiate a [`DiffusionPipeline`] from pretrained pipeline weights saved in the `.ckpt` format. The pipeline
        is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pretrained_model_link_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A link to the `.ckpt` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                    - A path to a *file* containing all pipeline weights.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If `"auto"` is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to True, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            extract_ema (`bool`, *optional*, defaults to `False`):
                Whether to extract the EMA weights or not. Pass `True` to extract the EMA weights which usually yield
                higher quality images for inference. Non-EMA weights are usually better to continue finetuning.
            upcast_attention (`bool`, *optional*, defaults to `None`):
                Whether the attention computation should always be upcasted.
            image_size (`int`, *optional*, defaults to 512):
                The image size the model was trained on. Use 512 for all Stable Diffusion v1 models and the Stable
                Diffusion v2 base model. Use 768 for Stable Diffusion v2.
            prediction_type (`str`, *optional*):
                The prediction type the model was trained on. Use `'epsilon'` for all Stable Diffusion v1 models and
                the Stable Diffusion v2 base model. Use `'v_prediction'` for Stable Diffusion v2.
            num_in_channels (`int`, *optional*, defaults to `None`):
                The number of input channels. If `None`, it will be automatically inferred.
            scheduler_type (`str`, *optional*, defaults to `"pndm"`):
                Type of scheduler to use. Should be one of `["pndm", "lms", "heun", "euler", "euler-ancestral", "dpm",
                "ddim"]`.
            load_safety_checker (`bool`, *optional*, defaults to `True`):
                Whether to load the safety checker or not.
            text_encoder (`CLIPTextModel`, *optional*, defaults to `None`):
                An instance of
                [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel) to use,
                specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
                variant. If this parameter is `None`, the function will load a new instance of [CLIP] by itself, if
                needed.
            tokenizer (`CLIPTokenizer`, *optional*, defaults to `None`):
                An instance of
                [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer)
                to use. If this parameter is `None`, the function will load a new instance of [CLIPTokenizer] by
                itself, if needed.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (for example the pipeline components of the
                specific pipeline class). The overwritten components are directly passed to the pipelines `__init__`
                method. See example below for more information.

        Examples:

        ```py
        >>> from diffusers import StableDiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = StableDiffusionPipeline.from_single_file(
        ...     "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors"
        ... )

        >>> # Download pipeline from local file
        >>> # file is downloaded under ./v1-5-pruned-emaonly.ckpt
        >>> pipeline = StableDiffusionPipeline.from_single_file("./v1-5-pruned-emaonly")

        >>> # Enable float16 and move to GPU
        >>> pipeline = StableDiffusionPipeline.from_single_file(
        ...     "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt",
        ...     torch_dtype=torch.float16,
        ... )
        >>> pipeline.to("cuda")
        ```
        """
        # import here to avoid circular dependency
        from .pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        extract_ema = kwargs.pop("extract_ema", False)
        image_size = kwargs.pop("image_size", None)
        scheduler_type = kwargs.pop("scheduler_type", "pndm")
        num_in_channels = kwargs.pop("num_in_channels", None)
        upcast_attention = kwargs.pop("upcast_attention", None)
        load_safety_checker = kwargs.pop("load_safety_checker", True)
        prediction_type = kwargs.pop("prediction_type", None)
        text_encoder = kwargs.pop("text_encoder", None)
        tokenizer = kwargs.pop("tokenizer", None)

        torch_dtype = kwargs.pop("torch_dtype", None)

        use_safetensors = kwargs.pop("use_safetensors", None if is_safetensors_available() else False)

        pipeline_name = cls.__name__
        file_extension = pretrained_model_link_or_path.rsplit(".", 1)[-1]
        from_safetensors = file_extension == "safetensors"

        if from_safetensors and use_safetensors is False:
            raise ValueError("Make sure to install `safetensors` with `pip install safetensors`.")

        # TODO: For now we only support stable diffusion
        stable_unclip = None
        model_type = None
        controlnet = False

        if pipeline_name == "StableDiffusionControlNetPipeline":
            # Model type will be inferred from the checkpoint.
            controlnet = True
        elif "StableDiffusion" in pipeline_name:
            # Model type will be inferred from the checkpoint.
            pass
        elif pipeline_name == "StableUnCLIPPipeline":
            model_type = "FrozenOpenCLIPEmbedder"
            stable_unclip = "txt2img"
        elif pipeline_name == "StableUnCLIPImg2ImgPipeline":
            model_type = "FrozenOpenCLIPEmbedder"
            stable_unclip = "img2img"
        elif pipeline_name == "PaintByExamplePipeline":
            model_type = "PaintByExample"
        elif pipeline_name == "LDMTextToImagePipeline":
            model_type = "LDMTextToImage"
        else:
            raise ValueError(f"Unhandled pipeline class: {pipeline_name}")

        # remove huggingface url
        for prefix in ["https://huggingface.co/", "huggingface.co/", "hf.co/", "https://hf.co/"]:
            if pretrained_model_link_or_path.startswith(prefix):
                pretrained_model_link_or_path = pretrained_model_link_or_path[len(prefix) :]

        # Code based on diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained
        ckpt_path = Path(pretrained_model_link_or_path)
        if not ckpt_path.is_file():
            # get repo_id and (potentially nested) file path of ckpt in repo
            repo_id = os.path.join(*ckpt_path.parts[:2])
            file_path = os.path.join(*ckpt_path.parts[2:])

            if file_path.startswith("blob/"):
                file_path = file_path[len("blob/") :]

            if file_path.startswith("main/"):
                file_path = file_path[len("main/") :]

            pretrained_model_link_or_path = hf_hub_download(
                repo_id,
                filename=file_path,
                cache_dir=cache_dir,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                force_download=force_download,
            )

        pipe = download_from_original_stable_diffusion_ckpt(
            pretrained_model_link_or_path,
            pipeline_class=cls,
            model_type=model_type,
            stable_unclip=stable_unclip,
            controlnet=controlnet,
            from_safetensors=from_safetensors,
            extract_ema=extract_ema,
            image_size=image_size,
            scheduler_type=scheduler_type,
            num_in_channels=num_in_channels,
            upcast_attention=upcast_attention,
            load_safety_checker=load_safety_checker,
            prediction_type=prediction_type,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        if torch_dtype is not None:
            pipe.to(torch_dtype=torch_dtype)

        return pipe
