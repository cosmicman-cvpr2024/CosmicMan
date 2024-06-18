from ..utils import (
    OptionalDependencyNotAvailable,
    is_flax_available,
    is_invisible_watermark_available,
    is_k_diffusion_available,
    is_librosa_available,
    is_note_seq_available,
    is_onnx_available,
    is_torch_available,
    is_transformers_available,
)


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_pt_objects import *  # noqa F403
else:
    from .consistency_models import ConsistencyModelPipeline
    from .dance_diffusion import DanceDiffusionPipeline
    from .ddim import DDIMPipeline
    from .ddpm import DDPMPipeline
    from .dit import DiTPipeline
    from .latent_diffusion import LDMSuperResolutionPipeline
    from .latent_diffusion_uncond import LDMPipeline
    from .pipeline_utils import AudioPipelineOutput, DiffusionPipeline, ImagePipelineOutput
    from .pndm import PNDMPipeline
    from .repaint import RePaintPipeline
    from .score_sde_ve import ScoreSdeVePipeline
    from .stochastic_karras_ve import KarrasVePipeline

try:
    if not (is_torch_available() and is_librosa_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_torch_and_librosa_objects import *  # noqa F403
else:
    from .audio_diffusion import AudioDiffusionPipeline, Mel

try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    from .alt_diffusion import AltDiffusionImg2ImgPipeline, AltDiffusionPipeline
    from .audioldm import AudioLDMPipeline
    from .controlnet import (
        StableDiffusionControlNetImg2ImgPipeline,
        StableDiffusionControlNetInpaintPipeline,
        StableDiffusionControlNetPipeline,
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLOurControlNetPipeline,
    )
    from .deepfloyd_if import (
        IFImg2ImgPipeline,
        IFImg2ImgSuperResolutionPipeline,
        IFInpaintingPipeline,
        IFInpaintingSuperResolutionPipeline,
        IFPipeline,
        IFSuperResolutionPipeline,
    )
    from .kandinsky import (
        KandinskyImg2ImgPipeline,
        KandinskyInpaintPipeline,
        KandinskyPipeline,
        KandinskyPriorPipeline,
    )
    from .kandinsky2_2 import (
        KandinskyV22ControlnetImg2ImgPipeline,
        KandinskyV22ControlnetPipeline,
        KandinskyV22Img2ImgPipeline,
        KandinskyV22InpaintPipeline,
        KandinskyV22Pipeline,
        KandinskyV22PriorEmb2EmbPipeline,
        KandinskyV22PriorPipeline,
    )
    from .latent_diffusion import LDMTextToImagePipeline
    from .paint_by_example import PaintByExamplePipeline
    from .semantic_stable_diffusion import SemanticStableDiffusionPipeline
    from .shap_e import ShapEImg2ImgPipeline, ShapEPipeline
    from .stable_diffusion import (
        CycleDiffusionPipeline,
        StableDiffusionAttendAndExcitePipeline,
        StableDiffusionDepth2ImgPipeline,
        StableDiffusionDiffEditPipeline,
        StableDiffusionImageVariationPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInpaintPipelineLegacy,
        StableDiffusionInstructPix2PixPipeline,
        StableDiffusionLatentUpscalePipeline,
        StableDiffusionLDM3DPipeline,
        StableDiffusionModelEditingPipeline,
        StableDiffusionPanoramaPipeline,
        StableDiffusionParadigmsPipeline,
        StableDiffusionPipeline,
        StableDiffusionPix2PixZeroPipeline,
        StableDiffusionSAGPipeline,
        StableDiffusionUpscalePipeline,
        StableUnCLIPImg2ImgPipeline,
        StableUnCLIPPipeline,
    )
    from .stable_diffusion_safe import StableDiffusionPipelineSafe
    from .t2i_adapter import StableDiffusionAdapterPipeline
    from .text_to_video_synthesis import TextToVideoSDPipeline, TextToVideoZeroPipeline, VideoToVideoSDPipeline
    from .unclip import UnCLIPImageVariationPipeline, UnCLIPPipeline
    from .unidiffuser import ImageTextPipelineOutput, UniDiffuserModel, UniDiffuserPipeline, UniDiffuserTextDecoder
    from .versatile_diffusion import (
        VersatileDiffusionDualGuidedPipeline,
        VersatileDiffusionImageVariationPipeline,
        VersatileDiffusionPipeline,
        VersatileDiffusionTextToImagePipeline,
    )
    from .vq_diffusion import VQDiffusionPipeline


try:
    if not (is_torch_available() and is_transformers_available() and is_invisible_watermark_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_torch_and_transformers_and_invisible_watermark_objects import *  # noqa F403
else:
    from .controlnet import StableDiffusionXLControlNetPipeline, StableDiffusionXLOurControlNetPipeline
    from .stable_diffusion_xl import (
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLPipeline,
        StableDiffusionXLImg2ImgPipeline,
    )

try:
    if not is_onnx_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_onnx_objects import *  # noqa F403
else:
    from .onnx_utils import OnnxRuntimeModel

try:
    if not (is_torch_available() and is_transformers_available() and is_onnx_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_torch_and_transformers_and_onnx_objects import *  # noqa F403
else:
    from .stable_diffusion import (
        OnnxStableDiffusionImg2ImgPipeline,
        OnnxStableDiffusionInpaintPipeline,
        OnnxStableDiffusionInpaintPipelineLegacy,
        OnnxStableDiffusionPipeline,
        OnnxStableDiffusionUpscalePipeline,
        StableDiffusionOnnxPipeline,
    )

try:
    if not (is_torch_available() and is_transformers_available() and is_k_diffusion_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_torch_and_transformers_and_k_diffusion_objects import *  # noqa F403
else:
    from .stable_diffusion import StableDiffusionKDiffusionPipeline

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_flax_objects import *  # noqa F403
else:
    from .pipeline_flax_utils import FlaxDiffusionPipeline


try:
    if not (is_flax_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_flax_and_transformers_objects import *  # noqa F403
else:
    from .controlnet import FlaxStableDiffusionControlNetPipeline
    from .stable_diffusion import (
        FlaxStableDiffusionImg2ImgPipeline,
        FlaxStableDiffusionInpaintPipeline,
        FlaxStableDiffusionPipeline,
    )
try:
    if not (is_transformers_available() and is_torch_available() and is_note_seq_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_transformers_and_torch_and_note_seq_objects import *  # noqa F403
else:
    from .spectrogram_diffusion import MidiProcessor, SpectrogramDiffusionPipeline
