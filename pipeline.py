"""
RGFVRPipeline — Reference-Guided Face Video Restoration.

Adapted from VideoX-Fun's Wan pipelines:
    https://github.com/aigc-apps/VideoX-Fun
"""

import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from PIL import Image

from wan import AutoencoderKLWan, AutoTokenizer, WanT5EncoderModel, WanRGFVRModel
from eva_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from helpers.restoration_utils import process_face_embeddings

logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        pass
        ```
"""


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class WanPipelineOutput(BaseOutput):
    videos: torch.Tensor


class RGFVRPipeline(DiffusionPipeline):
    """
    Reference face video restoration pipeline built on Wan2.1.

    Single flat class (no inheritance chain). Conditioning is guaranteed to use
    ArcFace (AntelopeV2) + EVA-CLIP embeddings from a single reference image.
    """

    _optional_components = ["text_encoder", "tokenizer"]
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "descriptive_prompt_embeds",
        "negative_descriptive_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: WanRGFVRModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
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
            scheduler=scheduler,
            face_main_model=face_main_model,
            face_helper_1=face_helper_1,
            face_helper_2=face_helper_2,
            face_clip_model=face_clip_model,
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae.spatial_compression_ratio,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

        eva_transform_mean = getattr(self.face_clip_model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(self.face_clip_model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            eva_transform_mean = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            eva_transform_std = (eva_transform_std,) * 3
        self.eva_transform_mean = eva_transform_mean
        self.eva_transform_std = eva_transform_std

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
            logger.warning(
                f"Input truncated to {max_sequence_length} tokens: {removed_text}"
            )

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
        negative_descriptive_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        descriptive_prompt_embeds: Optional[torch.Tensor] = None,
        negative_descriptive_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device

        descriptive_prompt = [descriptive_prompt] if isinstance(descriptive_prompt, str) else descriptive_prompt
        if descriptive_prompt is not None:
            batch_size = len(descriptive_prompt)
        else:
            batch_size = 1 if isinstance(descriptive_prompt_embeds, list) else descriptive_prompt_embeds.shape[0]

        if descriptive_prompt_embeds is None:
            descriptive_prompt_embeds = self._get_t5_prompt_embeds(
                descriptive_prompt=descriptive_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_descriptive_prompt_embeds is None:
            negative_descriptive_prompt = negative_descriptive_prompt or ""
            negative_descriptive_prompt = batch_size * [negative_descriptive_prompt] if isinstance(negative_descriptive_prompt, str) else negative_descriptive_prompt
            negative_descriptive_prompt_embeds = self._get_t5_prompt_embeds(
                descriptive_prompt=negative_descriptive_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return descriptive_prompt_embeds, negative_descriptive_prompt_embeds

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"Generators length {len(generator)} != batch size {batch_size}."
            )

        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            height // self.vae.spatial_compression_ratio,
            width // self.vae.spatial_compression_ratio,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        frames = frames.cpu().float().numpy()
        return frames

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        descriptive_prompt,
        height,
        width,
        negative_descriptive_prompt,
        callback_on_step_end_tensor_inputs,
        descriptive_prompt_embeds=None,
        negative_descriptive_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 8, got {height}, {width}.")
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` must be in {self._callback_tensor_inputs}."
            )
        if descriptive_prompt is not None and descriptive_prompt_embeds is not None:
            raise ValueError("Cannot forward both `descriptive_prompt` and `descriptive_prompt_embeds`.")
        if descriptive_prompt is None and descriptive_prompt_embeds is None:
            raise ValueError("Provide either `descriptive_prompt` or `descriptive_prompt_embeds`.")
        if descriptive_prompt is not None and not isinstance(descriptive_prompt, (str, list)):
            raise ValueError(f"`descriptive_prompt` must be str or list, got {type(descriptive_prompt)}.")

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    def _encode_control_frames(self, frames, ref_images=None, masks=None, vae=None):
        vae = self.vae if vae is None else vae
        weight_dtype = frames.dtype
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        if masks is None:
            latents = vae.encode(frames)[0].mode()
        else:
            masks = [torch.where(m > 0.5, 1.0, 0.0).to(weight_dtype) for m in masks]
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            inactive = vae.encode(inactive)[0].mode()
            reactive = vae.encode(reactive)[0].mode()
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]

        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                ref_latent = vae.encode(refs)[0].mode()
                if masks is not None:
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        return cat_latents

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

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        descriptive_prompt: Optional[Union[str, List[str]]] = None,
        negative_descriptive_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 512,
        degraded_video: Union[torch.FloatTensor] = None,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        descriptive_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_descriptive_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        reference_image: Union[Image.Image, str] = None,
        max_sequence_length: int = 512,
    ) -> Union[WanPipelineOutput, Tuple]:
        """
        Run restoration given a degraded video and a face reference image.

        Examples:

        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1

        self.check_inputs(
            descriptive_prompt, height, width, negative_descriptive_prompt,
            callback_on_step_end_tensor_inputs, descriptive_prompt_embeds, negative_descriptive_prompt_embeds,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        if descriptive_prompt is not None and isinstance(descriptive_prompt, str):
            batch_size = 1
        elif descriptive_prompt is not None and isinstance(descriptive_prompt, list):
            batch_size = len(descriptive_prompt)
        else:
            batch_size = 1 if isinstance(descriptive_prompt_embeds, list) else descriptive_prompt_embeds.shape[0]

        device = self._execution_device
        weight_dtype = self.vae.dtype

        do_classifier_free_guidance = guidance_scale > 1.0

        descriptive_prompt_embeds, negative_descriptive_prompt_embeds = self.encode_prompt(
            descriptive_prompt, negative_descriptive_prompt, do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            descriptive_prompt_embeds=descriptive_prompt_embeds,
            negative_descriptive_prompt_embeds=negative_descriptive_prompt_embeds,
            max_sequence_length=max_sequence_length, device=device,
        )
        if do_classifier_free_guidance:
            descriptive_features_input = negative_descriptive_prompt_embeds + descriptive_prompt_embeds
        else:
            descriptive_features_input = descriptive_prompt_embeds

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, mu=1
        )
        self._num_timesteps = len(timesteps)

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

        latent_channels = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt, latent_channels, num_frames,
            height, width, weight_dtype, device, generator, latents,
        )

        if degraded_video is not None:
            degraded_video_latents = self._encode_control_frames(degraded_video)

        if reference_image is not None:
            if isinstance(reference_image, str):
                reference_image_arr = np.array(Image.open(reference_image).convert("RGB"))
            else:
                reference_image_arr = np.array(reference_image.convert("RGB"))
            
            # global_perceptual_embedding: ArcFace & CLIP global features
            # local_perceptual_embeddings: CLIP local/patch-level features
            global_perceptual_embedding, local_perceptual_embeddings = self._process_face_embeddings(
                reference_image_arr, device=device, weight_dtype=weight_dtype
            )
            global_perceptual_embedding = global_perceptual_embedding.unsqueeze(0)
        else:
            global_perceptual_embedding = torch.zeros((1, 1280), device=device, dtype=weight_dtype).unsqueeze(0)
            local_perceptual_embeddings = torch.zeros((1, 577, 1024), device=device, dtype=weight_dtype)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

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

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self.transformer.num_inference_steps = num_inference_steps
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.transformer.current_steps = i
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if degraded_video is not None:
                    degraded_latents_input = torch.stack(degraded_video_latents * 2) if do_classifier_free_guidance else degraded_video_latents
                else:
                    degraded_latents_input = None

                timestep = t.expand(latent_model_input.shape[0])

                global_perceptual_embedding_input = (
                    torch.cat([torch.zeros_like(global_perceptual_embedding), global_perceptual_embedding])
                    if do_classifier_free_guidance else global_perceptual_embedding
                )
                local_perceptual_embeddings_input = (
                    torch.cat([torch.zeros_like(local_perceptual_embeddings), local_perceptual_embeddings])
                    if do_classifier_free_guidance else local_perceptual_embeddings
                )

                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                    noise_pred = self.transformer(
                        x=latent_model_input,
                        descriptive_features=descriptive_features_input,
                        t=timestep,
                        seq_len=seq_len,
                        y=degraded_latents_input,
                        perceptual_features=[global_perceptual_embedding_input, local_perceptual_embeddings_input],
                    )

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    descriptive_prompt_embeds = callback_outputs.pop("descriptive_prompt_embeds", descriptive_prompt_embeds)
                    negative_descriptive_prompt_embeds = callback_outputs.pop("negative_descriptive_prompt_embeds", negative_descriptive_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "numpy":
            video = self.decode_latents(latents)
        elif not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            video = torch.from_numpy(video)

        return WanPipelineOutput(videos=video)
