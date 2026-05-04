"""
RGFVRDMDPipeline — Reference-Guided Face Video Restoration (3-step DMD).

Mirrors RGFVRPipeline but replaces the 50-step flow-matching loop with a
3-step DMD (Distribution Matching Distillation) loop. No classifier-free
guidance is applied.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from PIL import Image

from wan import AutoencoderKLWan, AutoTokenizer, WanT5EncoderModel, WanRGFVRModel
from eva_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from helpers.restoration_utils import process_face_embeddings
from dmd.flow_match import FlowMatchScheduler

logger = logging.get_logger(__name__)

DENOISING_STEP_LIST = [1000, 757, 522]


@dataclass
class WanPipelineOutput(BaseOutput):
    videos: torch.Tensor


class RGFVRDMDPipeline(DiffusionPipeline):
    """
    3-step DMD pipeline for reference-guided face video restoration.

    Mirrors RGFVRPipeline but replaces the 50-step flow-matching denoising loop
    with a 3-step DMD loop. No classifier-free guidance is applied.
    """

    _optional_components = ["text_encoder", "tokenizer"]
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: WanRGFVRModel,
        face_main_model,
        face_helper_1,
        face_helper_2,
        face_clip_model,
    ):
        super(DiffusionPipeline, self).__init__()

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            face_main_model=face_main_model,
            face_helper_1=face_helper_1,
            face_helper_2=face_helper_2,
            face_clip_model=face_clip_model,
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)

        eva_transform_mean = getattr(self.face_clip_model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(self.face_clip_model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            eva_transform_mean = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            eva_transform_std = (eva_transform_std,) * 3
        self.eva_transform_mean = eva_transform_mean
        self.eva_transform_std = eva_transform_std

        # DMD scheduler — same shift/sigma_min as training
        self.dmd_scheduler = FlowMatchScheduler(
            shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.dmd_scheduler.set_timesteps(1000, training=False)

    def _get_t5_prompt_embeds(
        self,
        descriptive_prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        descriptive_prompt = [descriptive_prompt] if isinstance(descriptive_prompt, str) else descriptive_prompt
        batch_size = len(descriptive_prompt)

        text_inputs = self.tokenizer(
            descriptive_prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(descriptive_prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1:-1])
            logger.warning(f"Input truncated to {max_sequence_length} tokens: {removed_text}")

        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        descriptive_prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
        descriptive_prompt_embeds = descriptive_prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = descriptive_prompt_embeds.shape
        descriptive_prompt_embeds = descriptive_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        descriptive_prompt_embeds = descriptive_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return [u[:v] for u, v in zip(descriptive_prompt_embeds, seq_lens)]

    def encode_prompt(
        self,
        descriptive_prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        descriptive_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device

        descriptive_prompt = [descriptive_prompt] if isinstance(descriptive_prompt, str) else descriptive_prompt

        if descriptive_prompt_embeds is None:
            descriptive_prompt_embeds = self._get_t5_prompt_embeds(
                descriptive_prompt=descriptive_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return descriptive_prompt_embeds

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        frames = frames.cpu().float().numpy()
        return frames

    def check_inputs(
        self,
        descriptive_prompt,
        height,
        width,
        descriptive_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 8, got {height}, {width}.")
        if descriptive_prompt is not None and descriptive_prompt_embeds is not None:
            raise ValueError("Cannot forward both `descriptive_prompt` and `descriptive_prompt_embeds`.")
        if descriptive_prompt is None and descriptive_prompt_embeds is None:
            raise ValueError("Provide either `descriptive_prompt` or `descriptive_prompt_embeds`.")
        if descriptive_prompt is not None and not isinstance(descriptive_prompt, (str, list)):
            raise ValueError(f"`descriptive_prompt` must be str or list, got {type(descriptive_prompt)}.")

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    def _encode_control_frames(self, frames, vae=None):
        vae = self.vae if vae is None else vae
        latents = vae.encode(frames)[0].mode()
        return list(latents)  # list of B tensors [C, F_lat, H_lat, W_lat]

    def _process_face_embeddings(self, image, device, weight_dtype):
        return process_face_embeddings(
            image=image,
            app=self.face_main_model,
            face_helper_1=self.face_helper_1,
            face_helper_2=self.face_helper_2,
            clip_vision_model=self.face_clip_model,
            eva_transform_mean=self.eva_transform_mean,
            eva_transform_std=self.eva_transform_std,
            device=device,
            weight_dtype=weight_dtype,
        )

    def _get_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """Look up sigma for given timestep(s). Returns [B, 1, 1, 1, 1] for 5-D latent tensors."""
        sigmas = self.dmd_scheduler.sigmas.double().to(timestep.device)
        timesteps = self.dmd_scheduler.timesteps.double().to(timestep.device)
        t_id = torch.argmin((timesteps.unsqueeze(0) - timestep.double().unsqueeze(1)).abs(), dim=1)
        # Keep as float32 — timestep is torch.long; casting to it would truncate sigma to 0.
        return sigmas[t_id].float().reshape(-1, 1, 1, 1, 1)

    @torch.no_grad()
    def __call__(
        self,
        descriptive_prompt: Optional[Union[str, List[str]]] = None,
        negative_descriptive_prompt: Optional[Union[str, List[str]]] = None,  # unused; kept for interface parity
        height: int = 512,
        width: int = 512,
        degraded_video: Union[torch.FloatTensor] = None,
        num_frames: int = 81,
        num_inference_steps: int = 3,   # unused; DMD always runs len(DENOISING_STEP_LIST) steps
        guidance_scale: float = 1.0,    # unused; DMD has no CFG
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        descriptive_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        reference_image: Union[Image.Image, str] = None,
        max_sequence_length: int = 512,
    ) -> Union[WanPipelineOutput, Tuple]:
        """
        Run 3-step DMD restoration given a degraded video and a face reference image.
        """
        self.check_inputs(descriptive_prompt, height, width, descriptive_prompt_embeds)
        self._attention_kwargs = attention_kwargs

        batch_size = 1
        device = self._execution_device
        weight_dtype = self.vae.dtype

        # --- Text encoding ---
        descriptive_prompt_embeds = self.encode_prompt(
            descriptive_prompt,
            num_videos_per_prompt=1,
            descriptive_prompt_embeds=descriptive_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # --- Preprocess degraded video ---
        if degraded_video is not None:
            video_length = degraded_video.shape[2]
            degraded_video = self.image_processor.preprocess(
                rearrange(degraded_video, "b c f h w -> (b f) c h w"), height=height, width=width
            )
            degraded_video = degraded_video.to(dtype=torch.float32)
            degraded_video = rearrange(degraded_video, "(b f) c h w -> b c f h w", f=video_length)
            degraded_video = degraded_video.to(dtype=weight_dtype, device=device)
        else:
            video_length = num_frames

        if num_frames != video_length:
            num_frames = video_length

        # --- Encode control latents ---
        if degraded_video is not None:
            degraded_video_latents = self._encode_control_frames(degraded_video)
        else:
            degraded_video_latents = None

        # --- Reference face embeddings ---
        if reference_image is not None:
            if isinstance(reference_image, str):
                reference_image_arr = np.array(Image.open(reference_image).convert("RGB"))
            else:
                reference_image_arr = np.array(reference_image.convert("RGB"))
            global_perceptual_embedding, local_perceptual_embeddings = self._process_face_embeddings(
                reference_image_arr, device=device, weight_dtype=weight_dtype
            )
            global_perceptual_embedding = global_perceptual_embedding.unsqueeze(0)
        else:
            global_perceptual_embedding = torch.zeros((1, 1280), device=device, dtype=weight_dtype).unsqueeze(0)
            local_perceptual_embeddings = torch.zeros((1, 577, 1024), device=device, dtype=weight_dtype)

        # --- Compute seq_len for positional encoding ---
        target_shape = (
            self.vae.latent_channels,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            width // self.vae.spatial_compression_ratio,
            height // self.vae.spatial_compression_ratio,
        )
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3])
            / (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2])
            * target_shape[1]
        )

        # --- Initial noise [B, C, F_lat, H_lat, W_lat] ---
        latent_shape = (
            batch_size,
            self.vae.config.latent_channels,
            target_shape[1],
            target_shape[2],
            target_shape[3],
        )
        if latents is None:
            gen = generator[0] if isinstance(generator, list) else generator
            noisy = torch.randn(latent_shape, device=device, dtype=weight_dtype, generator=gen)
        else:
            noisy = latents.to(device)

        # --- 3-step DMD denoising loop ---
        with self.progress_bar(total=len(DENOISING_STEP_LIST)) as progress_bar:
            for step_idx, t_val in enumerate(DENOISING_STEP_LIST):
                t = torch.tensor([t_val] * batch_size, device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                    flow_pred = self.transformer(
                        x=noisy,
                        t=t,
                        descriptive_features=descriptive_prompt_embeds,
                        seq_len=seq_len,
                        y=degraded_video_latents,
                        perceptual_features=[global_perceptual_embedding, local_perceptual_embeddings],
                    )

                # Convert flow prediction to x0: x0 = noisy - sigma * flow_pred
                sigma = self._get_sigma(t)          # [B, 1, 1, 1, 1]
                x0 = noisy - sigma * flow_pred

                # Re-noise to next timestep (skip on last step)
                if step_idx < len(DENOISING_STEP_LIST) - 1:
                    next_t = torch.tensor(
                        [DENOISING_STEP_LIST[step_idx + 1]] * batch_size, device=device, dtype=torch.long
                    )
                    next_sigma = self._get_sigma(next_t)    # [B, 1, 1, 1, 1]
                    noise = torch.randn_like(x0)
                    noisy = (1 - next_sigma) * x0 + next_sigma * noise

                progress_bar.update()

        # --- Decode x0 ---
        if output_type == "numpy":
            video = self.decode_latents(x0)
        elif output_type != "latent":
            video = self.decode_latents(x0)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = x0

        self.maybe_free_model_hooks()

        if not return_dict:
            video = torch.from_numpy(video)

        return WanPipelineOutput(videos=video)
