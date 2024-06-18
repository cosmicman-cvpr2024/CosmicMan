__version__ = "0.19.0.dev0"

from .configuration_utils import ConfigMixin
from .utils import (
    OptionalDependencyNotAvailable,
    is_flax_available,
    is_inflect_available,
    is_invisible_watermark_available,
    is_k_diffusion_available,
    is_k_diffusion_version,
    is_librosa_available,
    is_note_seq_available,
    is_onnx_available,
    is_scipy_available,
    is_torch_available,
    is_torchsde_available,
    is_transformers_available,
    is_transformers_version,
    is_unidecode_available,
    logging,
)


try:
    if not is_onnx_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_onnx_objects import *  # noqa F403
else:
    from .pipelines import OnnxRuntimeModel

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_pt_objects import *  # noqa F403
else:
    from .models import (
        AutoencoderKL,
        ControlNetModel,
        ModelMixin,
        MultiAdapter,
        PriorTransformer,
        T2IAdapter,
        T5FilmDecoder,
        Transformer2DModel,
        UNet1DModel,
        UNet2DConditionModel,
        UNet2DConditionModel_New,
        UNet2DModel,
        UNet3DConditionModel,
        VQModel,
    )
    from .optimization import (
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
        get_scheduler,
    )
    from .pipelines import (
        AudioPipelineOutput,
        ConsistencyModelPipeline,
        DanceDiffusionPipeline,
        DDIMPipeline,
        DDPMPipeline,
        DiffusionPipeline,
        DiTPipeline,
        ImagePipelineOutput,
        KarrasVePipeline,
        LDMPipeline,
        LDMSuperResolutionPipeline,
        PNDMPipeline,
        RePaintPipeline,
        ScoreSdeVePipeline,
    )
    from .schedulers import (
        CMStochasticIterativeScheduler,
        DDIMInverseScheduler,
        DDIMParallelScheduler,
        DDIMScheduler,
        DDPMParallelScheduler,
        DDPMScheduler,
        DEISMultistepScheduler,
        DPMSolverMultistepInverseScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        HeunDiscreteScheduler,
        IPNDMScheduler,
        KarrasVeScheduler,
        KDPM2AncestralDiscreteScheduler,
        KDPM2DiscreteScheduler,
        PNDMScheduler,
        RePaintScheduler,
        SchedulerMixin,
        ScoreSdeVeScheduler,
        UnCLIPScheduler,
        UniPCMultistepScheduler,
        VQDiffusionScheduler,
    )
    from .training_utils import EMAModel

try:
    if not (is_torch_available() and is_scipy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_scipy_objects import *  # noqa F403
else:
    from .schedulers import LMSDiscreteScheduler

try:
    if not (is_torch_available() and is_torchsde_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_torchsde_objects import *  # noqa F403
else:
    from .schedulers import DPMSolverSDEScheduler

try:
    if not (is_torch_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_transformers_objects import *  # noqa F403
else:
    from .pipelines import (
        AltDiffusionImg2ImgPipeline,
        AltDiffusionPipeline,
        AudioLDMPipeline,
        CycleDiffusionPipeline,
        IFImg2ImgPipeline,
        IFImg2ImgSuperResolutionPipeline,
        IFInpaintingPipeline,
        IFInpaintingSuperResolutionPipeline,
        IFPipeline,
        IFSuperResolutionPipeline,
        ImageTextPipelineOutput,
        KandinskyImg2ImgPipeline,
        KandinskyInpaintPipeline,
        KandinskyPipeline,
        KandinskyPriorPipeline,
        KandinskyV22ControlnetImg2ImgPipeline,
        KandinskyV22ControlnetPipeline,
        KandinskyV22Img2ImgPipeline,
        KandinskyV22InpaintPipeline,
        KandinskyV22Pipeline,
        KandinskyV22PriorEmb2EmbPipeline,
        KandinskyV22PriorPipeline,
        LDMTextToImagePipeline,
        PaintByExamplePipeline,
        SemanticStableDiffusionPipeline,
        ShapEImg2ImgPipeline,
        ShapEPipeline,
        StableDiffusionAdapterPipeline,
        StableDiffusionAttendAndExcitePipeline,
        StableDiffusionControlNetImg2ImgPipeline,
        StableDiffusionControlNetInpaintPipeline,
        StableDiffusionControlNetPipeline,
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
        StableDiffusionPipelineSafe,
        StableDiffusionPix2PixZeroPipeline,
        StableDiffusionSAGPipeline,
        StableDiffusionUpscalePipeline,
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLOurControlNetPipeline,
        StableUnCLIPImg2ImgPipeline,
        StableUnCLIPPipeline,
        TextToVideoSDPipeline,
        TextToVideoZeroPipeline,
        UnCLIPImageVariationPipeline,
        UnCLIPPipeline,
        UniDiffuserModel,
        UniDiffuserPipeline,
        UniDiffuserTextDecoder,
        VersatileDiffusionDualGuidedPipeline,
        VersatileDiffusionImageVariationPipeline,
        VersatileDiffusionPipeline,
        VersatileDiffusionTextToImagePipeline,
        VideoToVideoSDPipeline,
        VQDiffusionPipeline,
    )

try:
    if not (is_torch_available() and is_transformers_available() and is_invisible_watermark_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_transformers_and_invisible_watermark_objects import *  # noqa F403
else:
    from .pipelines import (
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLOurControlNetPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLPipeline,
    )

try:
    if not (is_torch_available() and is_transformers_available() and is_k_diffusion_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_transformers_and_k_diffusion_objects import *  # noqa F403
else:
    from .pipelines import StableDiffusionKDiffusionPipeline

try:
    if not (is_torch_available() and is_transformers_available() and is_onnx_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_transformers_and_onnx_objects import *  # noqa F403
else:
    from .pipelines import (
        OnnxStableDiffusionImg2ImgPipeline,
        OnnxStableDiffusionInpaintPipeline,
        OnnxStableDiffusionInpaintPipelineLegacy,
        OnnxStableDiffusionPipeline,
        OnnxStableDiffusionUpscalePipeline,
        StableDiffusionOnnxPipeline,
    )

try:
    if not (is_torch_available() and is_librosa_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_torch_and_librosa_objects import *  # noqa F403
else:
    from .pipelines import AudioDiffusionPipeline, Mel

try:
    if not (is_transformers_available() and is_torch_available() and is_note_seq_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_transformers_and_torch_and_note_seq_objects import *  # noqa F403
else:
    from .pipelines import SpectrogramDiffusionPipeline

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_flax_objects import *  # noqa F403
else:
    from .models.controlnet_flax import FlaxControlNetModel
    from .models.modeling_flax_utils import FlaxModelMixin
    from .models.unet_2d_condition_flax import FlaxUNet2DConditionModel
    from .models.vae_flax import FlaxAutoencoderKL
    from .pipelines import FlaxDiffusionPipeline
    from .schedulers import (
        FlaxDDIMScheduler,
        FlaxDDPMScheduler,
        FlaxDPMSolverMultistepScheduler,
        FlaxKarrasVeScheduler,
        FlaxLMSDiscreteScheduler,
        FlaxPNDMScheduler,
        FlaxSchedulerMixin,
        FlaxScoreSdeVeScheduler,
    )


try:
    if not (is_flax_available() and is_transformers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_flax_and_transformers_objects import *  # noqa F403
else:
    from .pipelines import (
        FlaxStableDiffusionControlNetPipeline,
        FlaxStableDiffusionImg2ImgPipeline,
        FlaxStableDiffusionInpaintPipeline,
        FlaxStableDiffusionPipeline,
    )

try:
    if not (is_note_seq_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils.dummy_note_seq_objects import *  # noqa F403
else:
    from .pipelines import MidiProcessor
