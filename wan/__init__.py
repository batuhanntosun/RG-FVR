"""
Wan2.1 model wrappers for the RFVR standalone pipeline.

Adapted from VideoX-Fun:
    https://github.com/aigc-apps/VideoX-Fun
"""
from diffusers import AutoencoderKL
from transformers import (AutoTokenizer, CLIPImageProcessor, CLIPTextModel,
                          CLIPTokenizer, CLIPVisionModelWithProjection,
                          T5EncoderModel, T5Tokenizer, T5TokenizerFast)

from .wan_image_encoder import CLIPModel
from .wan_text_encoder import WanT5EncoderModel
from .wan_transformer3d import (WanRMSNorm,
                                WanSelfAttention, WanRGFVRModel)
from .wan_vae import AutoencoderKLWan
