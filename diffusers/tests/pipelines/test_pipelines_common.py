import contextlib
import gc
import inspect
import io
import re
import tempfile
import unittest
from typing import Callable, Union

import numpy as np
import PIL
import torch

import diffusers
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging
from diffusers.utils.import_utils import is_accelerate_available, is_accelerate_version, is_xformers_available
from diffusers.utils.testing_utils import CaptureLogger, require_torch, torch_device


def to_np(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    return tensor


def check_same_shape(tensor_list):
    shapes = [tensor.shape for tensor in tensor_list]
    return all(shape == shapes[0] for shape in shapes[1:])


class PipelineLatentTesterMixin:
    """
    This mixin is designed to be used with PipelineTesterMixin and unittest.TestCase classes.
    It provides a set of common tests for PyTorch pipeline that has vae, e.g.
    equivalence of different input and output types, etc.
    """

    @property
    def image_params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `image_params` in the child test class. "
            "`image_params` are tested for if all accepted input image types (i.e. `pt`,`pil`,`np`) are producing same results"
        )

    @property
    def image_latents_params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `image_latents_params` in the child test class. "
            "`image_latents_params` are tested for if passing latents directly are producing same results"
        )

    def get_dummy_inputs_by_type(self, device, seed=0, input_image_type="pt", output_type="np"):
        inputs = self.get_dummy_inputs(device, seed)

        def convert_to_pt(image):
            if isinstance(image, torch.Tensor):
                input_image = image
            elif isinstance(image, np.ndarray):
                input_image = VaeImageProcessor.numpy_to_pt(image)
            elif isinstance(image, PIL.Image.Image):
                input_image = VaeImageProcessor.pil_to_numpy(image)
                input_image = VaeImageProcessor.numpy_to_pt(input_image)
            else:
                raise ValueError(f"unsupported input_image_type {type(image)}")
            return input_image

        def convert_pt_to_type(image, input_image_type):
            if input_image_type == "pt":
                input_image = image
            elif input_image_type == "np":
                input_image = VaeImageProcessor.pt_to_numpy(image)
            elif input_image_type == "pil":
                input_image = VaeImageProcessor.pt_to_numpy(image)
                input_image = VaeImageProcessor.numpy_to_pil(input_image)
            else:
                raise ValueError(f"unsupported input_image_type {input_image_type}.")
            return input_image

        for image_param in self.image_params:
            if image_param in inputs.keys():
                inputs[image_param] = convert_pt_to_type(
                    convert_to_pt(inputs[image_param]).to(device), input_image_type
                )

        inputs["output_type"] = output_type

        return inputs

    def test_pt_np_pil_outputs_equivalent(self, expected_max_diff=1e-4):
        self._test_pt_np_pil_outputs_equivalent(expected_max_diff=expected_max_diff)

    def _test_pt_np_pil_outputs_equivalent(self, expected_max_diff=1e-4, input_image_type="pt"):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        output_pt = pipe(
            **self.get_dummy_inputs_by_type(torch_device, input_image_type=input_image_type, output_type="pt")
        )[0]
        output_np = pipe(
            **self.get_dummy_inputs_by_type(torch_device, input_image_type=input_image_type, output_type="np")
        )[0]
        output_pil = pipe(
            **self.get_dummy_inputs_by_type(torch_device, input_image_type=input_image_type, output_type="pil")
        )[0]

        max_diff = np.abs(output_pt.cpu().numpy().transpose(0, 2, 3, 1) - output_np).max()
        self.assertLess(
            max_diff, expected_max_diff, "`output_type=='pt'` generate different results from `output_type=='np'`"
        )

        max_diff = np.abs(np.array(output_pil[0]) - (output_np * 255).round()).max()
        self.assertLess(max_diff, 2.0, "`output_type=='pil'` generate different results from `output_type=='np'`")

    def test_pt_np_pil_inputs_equivalent(self):
        if len(self.image_params) == 0:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        out_input_pt = pipe(**self.get_dummy_inputs_by_type(torch_device, input_image_type="pt"))[0]
        out_input_np = pipe(**self.get_dummy_inputs_by_type(torch_device, input_image_type="np"))[0]
        out_input_pil = pipe(**self.get_dummy_inputs_by_type(torch_device, input_image_type="pil"))[0]

        max_diff = np.abs(out_input_pt - out_input_np).max()
        self.assertLess(max_diff, 1e-4, "`input_type=='pt'` generate different result from `input_type=='np'`")
        max_diff = np.abs(out_input_pil - out_input_np).max()
        self.assertLess(max_diff, 1e-2, "`input_type=='pt'` generate different result from `input_type=='np'`")

    def test_latents_input(self):
        if len(self.image_latents_params) == 0:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.image_processor = VaeImageProcessor(do_resize=False, do_normalize=False)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        out = pipe(**self.get_dummy_inputs_by_type(torch_device, input_image_type="pt"))[0]

        vae = components["vae"]
        inputs = self.get_dummy_inputs_by_type(torch_device, input_image_type="pt")
        generator = inputs["generator"]
        for image_param in self.image_latents_params:
            if image_param in inputs.keys():
                inputs[image_param] = (
                    vae.encode(inputs[image_param]).latent_dist.sample(generator) * vae.config.scaling_factor
                )
        out_latents_inputs = pipe(**inputs)[0]

        max_diff = np.abs(out - out_latents_inputs).max()
        self.assertLess(max_diff, 1e-4, "passing latents as image input generate different result from passing image")


@require_torch
class PipelineKarrasSchedulerTesterMixin:
    """
    This mixin is designed to be used with unittest.TestCase classes.
    It provides a set of common tests for each PyTorch pipeline that makes use of KarrasDiffusionSchedulers
    equivalence of dict and tuple outputs, etc.
    """

    def test_karras_schedulers_shape(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)

        # make sure that PNDM does not need warm-up
        pipe.scheduler.register_to_config(skip_prk_steps=True)

        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_inference_steps"] = 2

        if "strength" in inputs:
            inputs["num_inference_steps"] = 4
            inputs["strength"] = 0.5

        outputs = []
        for scheduler_enum in KarrasDiffusionSchedulers:
            if "KDPM2" in scheduler_enum.name:
                inputs["num_inference_steps"] = 5

            scheduler_cls = getattr(diffusers, scheduler_enum.name)
            pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
            output = pipe(**inputs)[0]
            outputs.append(output)

            if "KDPM2" in scheduler_enum.name:
                inputs["num_inference_steps"] = 2

        assert check_same_shape(outputs)


@require_torch
class PipelineTesterMixin:
    """
    This mixin is designed to be used with unittest.TestCase classes.
    It provides a set of common tests for each PyTorch pipeline, e.g. saving and loading the pipeline,
    equivalence of dict and tuple outputs, etc.
    """

    # Canonical parameters that are passed to `__call__` regardless
    # of the type of pipeline. They are always optional and have common
    # sense default values.
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "num_images_per_prompt",
            "generator",
            "latents",
            "output_type",
            "return_dict",
            "callback",
            "callback_steps",
        ]
    )

    # set these parameters to False in the child class if the pipeline does not support the corresponding functionality
    test_attention_slicing = True

    test_xformers_attention = True

    def get_generator(self, seed):
        device = torch_device if torch_device != "mps" else "cpu"
        generator = torch.Generator(device).manual_seed(seed)
        return generator

    @property
    def pipeline_class(self) -> Union[Callable, DiffusionPipeline]:
        raise NotImplementedError(
            "You need to set the attribute `pipeline_class = ClassNameOfPipeline` in the child test class. "
            "See existing pipeline tests for reference."
        )

    def get_dummy_components(self):
        raise NotImplementedError(
            "You need to implement `get_dummy_components(self)` in the child test class. "
            "See existing pipeline tests for reference."
        )

    def get_dummy_inputs(self, device, seed=0):
        raise NotImplementedError(
            "You need to implement `get_dummy_inputs(self, device, seed)` in the child test class. "
            "See existing pipeline tests for reference."
        )

    @property
    def params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `params` in the child test class. "
            "`params` are checked for if all values are present in `__call__`'s signature."
            " You can set `params` using one of the common set of parameters defined in `pipeline_params.py`"
            " e.g., `TEXT_TO_IMAGE_PARAMS` defines the common parameters used in text to  "
            "image pipelines, including prompts and prompt embedding overrides."
            "If your pipeline's set of arguments has minor changes from one of the common sets of arguments, "
            "do not make modifications to the existing common sets of arguments. I.e. a text to image pipeline "
            "with non-configurable height and width arguments should set the attribute as "
            "`params = TEXT_TO_IMAGE_PARAMS - {'height', 'width'}`. "
            "See existing pipeline tests for reference."
        )

    @property
    def batch_params(self) -> frozenset:
        raise NotImplementedError(
            "You need to set the attribute `batch_params` in the child test class. "
            "`batch_params` are the parameters required to be batched when passed to the pipeline's "
            "`__call__` method. `pipeline_params.py` provides some common sets of parameters such as "
            "`TEXT_TO_IMAGE_BATCH_PARAMS`, `IMAGE_VARIATION_BATCH_PARAMS`, etc... If your pipeline's "
            "set of batch arguments has minor changes from one of the common sets of batch arguments, "
            "do not make modifications to the existing common sets of batch arguments. I.e. a text to "
            "image pipeline `negative_prompt` is not batched should set the attribute as "
            "`batch_params = TEXT_TO_IMAGE_BATCH_PARAMS - {'negative_prompt'}`. "
            "See existing pipeline tests for reference."
        )

    def tearDown(self):
        # clean up the VRAM after each test in case of CUDA runtime errors
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_save_load_local(self, expected_max_difference=1e-4):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0]

        logger = logging.get_logger("diffusers.pipelines.pipeline_utils")
        logger.setLevel(diffusers.logging.INFO)

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)

            with CaptureLogger(logger) as cap_logger:
                pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)

            for name in pipe_loaded.components.keys():
                if name not in pipe_loaded._optional_components:
                    assert name in str(cap_logger)

            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(max_diff, expected_max_difference)

    def test_pipeline_call_signature(self):
        self.assertTrue(
            hasattr(self.pipeline_class, "__call__"), f"{self.pipeline_class} should have a `__call__` method"
        )

        parameters = inspect.signature(self.pipeline_class.__call__).parameters

        optional_parameters = set()

        for k, v in parameters.items():
            if v.default != inspect._empty:
                optional_parameters.add(k)

        parameters = set(parameters.keys())
        parameters.remove("self")
        parameters.discard("kwargs")  # kwargs can be added if arguments of pipeline call function are deprecated

        remaining_required_parameters = set()

        for param in self.params:
            if param not in parameters:
                remaining_required_parameters.add(param)

        self.assertTrue(
            len(remaining_required_parameters) == 0,
            f"Required parameters not present: {remaining_required_parameters}",
        )

        remaining_required_optional_parameters = set()

        for param in self.required_optional_params:
            if param not in optional_parameters:
                remaining_required_optional_parameters.add(param)

        self.assertTrue(
            len(remaining_required_optional_parameters) == 0,
            f"Required optional parameters not present: {remaining_required_optional_parameters}",
        )

    def test_inference_batch_consistent(self, batch_sizes=[2, 4, 13]):
        self._test_inference_batch_consistent(batch_sizes=batch_sizes)

    def _test_inference_batch_consistent(
        self, batch_sizes=[2, 4, 13], additional_params_copy_to_batched_inputs=["num_inference_steps"]
    ):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)

        logger = logging.get_logger(pipe.__module__)
        logger.setLevel(level=diffusers.logging.FATAL)

        # batchify inputs
        for batch_size in batch_sizes:
            batched_inputs = {}
            for name, value in inputs.items():
                if name in self.batch_params:
                    # prompt is string
                    if name == "prompt":
                        len_prompt = len(value)
                        # make unequal batch sizes
                        batched_inputs[name] = [value[: len_prompt // i] for i in range(1, batch_size + 1)]

                        # make last batch super long
                        batched_inputs[name][-1] = 2000 * "very long"
                    # or else we have images
                    else:
                        batched_inputs[name] = batch_size * [value]
                elif name == "batch_size":
                    batched_inputs[name] = batch_size
                else:
                    batched_inputs[name] = value

            for arg in additional_params_copy_to_batched_inputs:
                batched_inputs[arg] = inputs[arg]

            batched_inputs["output_type"] = "np"

            if self.pipeline_class.__name__ == "DanceDiffusionPipeline":
                batched_inputs.pop("output_type")

            output = pipe(**batched_inputs)

            assert len(output[0]) == batch_size

            batched_inputs["output_type"] = "np"

            if self.pipeline_class.__name__ == "DanceDiffusionPipeline":
                batched_inputs.pop("output_type")

            output = pipe(**batched_inputs)[0]

            assert output.shape[0] == batch_size

        logger.setLevel(level=diffusers.logging.WARNING)

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-4):
        self._test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def _test_inference_batch_single_identical(
        self,
        batch_size=3,
        test_max_difference=None,
        test_mean_pixel_difference=None,
        relax_max_difference=False,
        expected_max_diff=1e-4,
        additional_params_copy_to_batched_inputs=["num_inference_steps"],
    ):
        if test_max_difference is None:
            # TODO(Pedro) - not sure why, but not at all reproducible at the moment it seems
            # make sure that batched and non-batched is identical
            test_max_difference = torch_device != "mps"

        if test_mean_pixel_difference is None:
            # TODO same as above
            test_mean_pixel_difference = torch_device != "mps"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)

        logger = logging.get_logger(pipe.__module__)
        logger.setLevel(level=diffusers.logging.FATAL)

        # batchify inputs
        batched_inputs = {}
        batch_size = batch_size
        for name, value in inputs.items():
            if name in self.batch_params:
                # prompt is string
                if name == "prompt":
                    len_prompt = len(value)
                    # make unequal batch sizes
                    batched_inputs[name] = [value[: len_prompt // i] for i in range(1, batch_size + 1)]

                    # make last batch super long
                    batched_inputs[name][-1] = 2000 * "very long"
                # or else we have images
                else:
                    batched_inputs[name] = batch_size * [value]
            elif name == "batch_size":
                batched_inputs[name] = batch_size
            elif name == "generator":
                batched_inputs[name] = [self.get_generator(i) for i in range(batch_size)]
            else:
                batched_inputs[name] = value

        for arg in additional_params_copy_to_batched_inputs:
            batched_inputs[arg] = inputs[arg]

        if self.pipeline_class.__name__ != "DanceDiffusionPipeline":
            batched_inputs["output_type"] = "np"

        output_batch = pipe(**batched_inputs)
        assert output_batch[0].shape[0] == batch_size

        inputs["generator"] = self.get_generator(0)

        output = pipe(**inputs)

        logger.setLevel(level=diffusers.logging.WARNING)
        if test_max_difference:
            if relax_max_difference:
                # Taking the median of the largest <n> differences
                # is resilient to outliers
                diff = np.abs(output_batch[0][0] - output[0][0])
                diff = diff.flatten()
                diff.sort()
                max_diff = np.median(diff[-5:])
            else:
                max_diff = np.abs(output_batch[0][0] - output[0][0]).max()
            assert max_diff < expected_max_diff

        if test_mean_pixel_difference:
            assert_mean_pixel_difference(output_batch[0][0], output[0][0])

    def test_dict_tuple_outputs_equivalent(self, expected_max_difference=1e-4):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs(torch_device))[0]
        output_tuple = pipe(**self.get_dummy_inputs(torch_device), return_dict=False)[0]

        max_diff = np.abs(to_np(output) - to_np(output_tuple)).max()
        self.assertLess(max_diff, expected_max_difference)

    def test_components_function(self):
        init_components = self.get_dummy_components()
        pipe = self.pipeline_class(**init_components)

        self.assertTrue(hasattr(pipe, "components"))
        self.assertTrue(set(pipe.components.keys()) == set(init_components.keys()))

    @unittest.skipIf(torch_device != "cuda", reason="float16 requires CUDA")
    def test_float16_inference(self, expected_max_diff=1e-2):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        pipe_fp16 = self.pipeline_class(**components)
        pipe_fp16.to(torch_device, torch.float16)
        pipe_fp16.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs(torch_device))[0]
        output_fp16 = pipe_fp16(**self.get_dummy_inputs(torch_device))[0]

        max_diff = np.abs(to_np(output) - to_np(output_fp16)).max()
        self.assertLess(max_diff, expected_max_diff, "The outputs of the fp16 and fp32 pipelines are too different.")

    @unittest.skipIf(torch_device != "cuda", reason="float16 requires CUDA")
    def test_save_load_float16(self, expected_max_diff=1e-2):
        components = self.get_dummy_components()
        for name, module in components.items():
            if hasattr(module, "half"):
                components[name] = module.to(torch_device).half()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir, torch_dtype=torch.float16)
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        for name, component in pipe_loaded.components.items():
            if hasattr(component, "dtype"):
                self.assertTrue(
                    component.dtype == torch.float16,
                    f"`{name}.dtype` switched from `float16` to {component.dtype} after loading.",
                )

        inputs = self.get_dummy_inputs(torch_device)
        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(
            max_diff, expected_max_diff, "The output of the fp16 pipeline changed after saving and loading."
        )

    def test_save_load_optional_components(self, expected_max_difference=1e-4):
        if not hasattr(self.pipeline_class, "_optional_components"):
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        # set all optional components to None
        for optional_component in pipe._optional_components:
            setattr(pipe, optional_component, None)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            pipe_loaded.to(torch_device)
            pipe_loaded.set_progress_bar_config(disable=None)

        for optional_component in pipe._optional_components:
            self.assertTrue(
                getattr(pipe_loaded, optional_component) is None,
                f"`{optional_component}` did not stay set to None after loading.",
            )

        inputs = self.get_dummy_inputs(torch_device)
        output_loaded = pipe_loaded(**inputs)[0]

        max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
        self.assertLess(max_diff, expected_max_difference)

    @unittest.skipIf(torch_device != "cuda", reason="CUDA and CPU are required to switch devices")
    def test_to_device(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        pipe.to("cpu")
        model_devices = [component.device.type for component in components.values() if hasattr(component, "device")]
        self.assertTrue(all(device == "cpu" for device in model_devices))

        output_cpu = pipe(**self.get_dummy_inputs("cpu"))[0]
        self.assertTrue(np.isnan(output_cpu).sum() == 0)

        pipe.to("cuda")
        model_devices = [component.device.type for component in components.values() if hasattr(component, "device")]
        self.assertTrue(all(device == "cuda" for device in model_devices))

        output_cuda = pipe(**self.get_dummy_inputs("cuda"))[0]
        self.assertTrue(np.isnan(to_np(output_cuda)).sum() == 0)

    def test_to_dtype(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)

        model_dtypes = [component.dtype for component in components.values() if hasattr(component, "dtype")]
        self.assertTrue(all(dtype == torch.float32 for dtype in model_dtypes))

        pipe.to(torch_dtype=torch.float16)
        model_dtypes = [component.dtype for component in components.values() if hasattr(component, "dtype")]
        self.assertTrue(all(dtype == torch.float16 for dtype in model_dtypes))

    def test_attention_slicing_forward_pass(self, expected_max_diff=1e-3):
        self._test_attention_slicing_forward_pass(expected_max_diff=expected_max_diff)

    def _test_attention_slicing_forward_pass(
        self, test_max_difference=True, test_mean_pixel_difference=True, expected_max_diff=1e-3
    ):
        if not self.test_attention_slicing:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_without_slicing = pipe(**inputs)[0]

        pipe.enable_attention_slicing(slice_size=1)
        inputs = self.get_dummy_inputs(torch_device)
        output_with_slicing = pipe(**inputs)[0]

        if test_max_difference:
            max_diff = np.abs(to_np(output_with_slicing) - to_np(output_without_slicing)).max()
            self.assertLess(max_diff, expected_max_diff, "Attention slicing should not affect the inference results")

        if test_mean_pixel_difference:
            assert_mean_pixel_difference(output_with_slicing[0], output_without_slicing[0])

    @unittest.skipIf(
        torch_device != "cuda" or not is_accelerate_available() or is_accelerate_version("<", "0.14.0"),
        reason="CPU offload is only available with CUDA and `accelerate v0.14.0` or higher",
    )
    def test_cpu_offload_forward_pass(self, expected_max_diff=1e-4):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_without_offload = pipe(**inputs)[0]

        pipe.enable_sequential_cpu_offload()
        inputs = self.get_dummy_inputs(torch_device)
        output_with_offload = pipe(**inputs)[0]

        max_diff = np.abs(to_np(output_with_offload) - to_np(output_without_offload)).max()
        self.assertLess(max_diff, expected_max_diff, "CPU offloading should not affect the inference results")

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass()

    def _test_xformers_attention_forwardGenerator_pass(
        self, test_max_difference=True, test_mean_pixel_difference=True, expected_max_diff=1e-4
    ):
        if not self.test_xformers_attention:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output_without_offload = pipe(**inputs)[0]
        output_without_offload = (
            output_without_offload.cpu() if torch.is_tensor(output_without_offload) else output_without_offload
        )

        pipe.enable_xformers_memory_efficient_attention()
        inputs = self.get_dummy_inputs(torch_device)
        output_with_offload = pipe(**inputs)[0]
        output_with_offload = (
            output_with_offload.cpu() if torch.is_tensor(output_with_offload) else output_without_offload
        )

        if test_max_difference:
            max_diff = np.abs(output_with_offload - output_without_offload).max()
            self.assertLess(max_diff, expected_max_diff, "XFormers attention should not affect the inference results")

        if test_mean_pixel_difference:
            assert_mean_pixel_difference(output_with_offload[0], output_without_offload[0])

    def test_progress_bar(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)

        inputs = self.get_dummy_inputs(torch_device)
        with io.StringIO() as stderr, contextlib.redirect_stderr(stderr):
            _ = pipe(**inputs)
            stderr = stderr.getvalue()
            # we can't calculate the number of progress steps beforehand e.g. for strength-dependent img2img,
            # so we just match "5" in "#####| 1/5 [00:01<00:00]"
            max_steps = re.search("/(.*?) ", stderr).group(1)
            self.assertTrue(max_steps is not None and len(max_steps) > 0)
            self.assertTrue(
                f"{max_steps}/{max_steps}" in stderr, "Progress bar should be enabled and stopped at the max step"
            )

        pipe.set_progress_bar_config(disable=True)
        with io.StringIO() as stderr, contextlib.redirect_stderr(stderr):
            _ = pipe(**inputs)
            self.assertTrue(stderr.getvalue() == "", "Progress bar should be disabled")

    def test_num_images_per_prompt(self):
        sig = inspect.signature(self.pipeline_class.__call__)

        if "num_images_per_prompt" not in sig.parameters:
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        batch_sizes = [1, 2]
        num_images_per_prompts = [1, 2]

        for batch_size in batch_sizes:
            for num_images_per_prompt in num_images_per_prompts:
                inputs = self.get_dummy_inputs(torch_device)

                for key in inputs.keys():
                    if key in self.batch_params:
                        inputs[key] = batch_size * [inputs[key]]

                images = pipe(**inputs, num_images_per_prompt=num_images_per_prompt)[0]

                assert images.shape[0] == batch_size * num_images_per_prompt


# Some models (e.g. unCLIP) are extremely likely to significantly deviate depending on which hardware is used.
# This helper function is used to check that the image doesn't deviate on average more than 10 pixels from a
# reference image.
def assert_mean_pixel_difference(image, expected_image, expected_max_diff=10):
    image = np.asarray(DiffusionPipeline.numpy_to_pil(image)[0], dtype=np.float32)
    expected_image = np.asarray(DiffusionPipeline.numpy_to_pil(expected_image)[0], dtype=np.float32)
    avg_diff = np.abs(image - expected_image).mean()
    assert avg_diff < expected_max_diff, f"Error image deviates {avg_diff} pixels on average"
