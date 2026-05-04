import os
import json
from glob import glob
import shutil
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from torchvision.transforms.functional import resize
from diffusers import FlowMatchEulerDiscreteScheduler
from decord import cpu, VideoReader
import av
import cv2
from PIL import Image
from tqdm import tqdm
import gc
import time

import sys
from pathlib import Path
# Make the standalone dir itself importable so inner packages resolve with short names.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wan import (
    AutoencoderKLWan,
    WanT5EncoderModel,
    AutoTokenizer,
    WanRGFVRModel,
)
from helpers.restoration_utils import (
    prepare_face_models,
    prepare_face_attr_model,
    process_face_attributes,
)
from pipeline import RGFVRPipeline

# Processing Settings
N_FRAMES = 81 # Set integer (e.g. 21) or None for all frames

# All model weights and metadata are expected under a single checkpoints directory
# (default: standalone/ckpts). Override with --ckpts_dir. Expected layout:
#   <ckpts_dir>/config.yaml
#   <ckpts_dir>/transformer/           (restorer)
#   <ckpts_dir>/vae/Wan2.1_VAE.pth
#   <ckpts_dir>/text_encoder/
#   <ckpts_dir>/tokenizer/
#   <ckpts_dir>/scheduler/
#   <ckpts_dir>/face_encoder/          (prepare_face_models reads `face_encoder/` subdir inside ckpts_dir)
#   <ckpts_dir>/farl/
#   <ckpts_dir>/arcface/
#   <ckpts_dir>/metadata/meta_info.json
#   <ckpts_dir>/metadata/appearance_categories.json
DEFAULT_CKPTS_DIR = str(Path(__file__).resolve().parent / "ckpts")

# Inference Settings
USE_REFERENCE = True
NUM_INFERENCE_STEPS = 50

# Descriptive Text Settings
PROB_TH = 0.6
START_PHRASE = "A photorealistic, high-definition close-up video of a "
END_PHRASE = ", 4k resolution, high temporal consistency, sharp facial details, realistic skin texture."
DEFAULT_PROMPT = "A photorealistic, high-definition close-up video of a person 4k resolution, high temporal consistency, sharp facial details, realistic skin texture."

# Degree -> allowed categories (degree 0 means "no descriptive text at all")
DEGREE_TO_CATEGORIES = {
    1: ["gender"], 
    2: ["gender", "hair",],
    3: ["gender", "hair", "facial_hair"],
    4: ["gender", "hair", "facial_hair", "accessories_and_makeup"],
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DTYPE = torch.bfloat16
SEED = 42

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "RGFVR: Reference-Guided Face Video Restoration. "
            "Restores a degraded face video using a high-quality reference image. "
            "Input video should be 512x512; optimal length is 81 frames. "
            "If the reference is loosely aligned with the subject, try a lower --degree."
        )
    )
    parser.add_argument("--video_path", type=str, default="./video.mp4",
                        help="Degraded input video (512x512 recommended). Default: ./video.mp4")
    parser.add_argument("--reference_path", type=str, default="./ref.jpg",
                        help="High-quality reference image of the target subject. Default: ./ref.jpg")
    parser.add_argument("--result_path", type=str, default=".",
                        help="Output directory. Default: current directory")
    parser.add_argument("--degree", type=int, default=1, choices=[0, 1, 2, 3, 4],
                        help=(
                            "Descriptive prompt detail level derived from the reference image. "
                            "0: generic prompt. "
                            "1: gender. "
                            "2: gender + hair. "
                            "3: gender + hair + facial hair. "
                            "4: all attributes (default). "
                            "Lower values are more robust when the reference is loosely aligned."
                        ))
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom text prompt. Overrides --degree when provided.")
    parser.add_argument("--guidance_scale", type=float, default=2.0,
                        help="Classifier-free guidance scale (default: 2.0).")
    parser.add_argument("--save_frames", action="store_true",
                        help="If set, also save each output frame as a PNG next to the video.")
    parser.add_argument("--fps", type=float, default=None,
                        help="FPS for the output video. Default: match the input video's fps.")
    parser.add_argument("--ckpts_dir", type=str, default=DEFAULT_CKPTS_DIR,
                        help=(
                            "Directory containing all model weights and metadata. "
                            f"Default: {DEFAULT_CKPTS_DIR}. "
                            "Expected subfolders: transformer/, vae/, text_encoder/, tokenizer/, "
                            "scheduler/, face_encoder/, farl/, arcface/, metadata/ and a config.yaml."
                        ))
    return parser.parse_args()


def resolve_ckpt_paths(ckpts_dir):
    """Return a dict of all paths derived from the single ckpts_dir."""
    ckpts_dir = os.path.abspath(ckpts_dir)
    return {
        "ckpts_dir": ckpts_dir,
        "config": os.path.join(ckpts_dir, "config.yaml"),
        "model_root": ckpts_dir,                           # tokenizer / vae / text_encoder subpaths resolve here
        "transformer": os.path.join(ckpts_dir, "transformer"),
        "face_encoder_root": ckpts_dir,                    # prepare_face_models reads `<root>/face_encoder/*`
        "farl": os.path.join(ckpts_dir, "farl"),
        "meta_info": os.path.join(ckpts_dir, "metadata", "meta_info.json"),
        "appearance_categories": os.path.join(ckpts_dir, "metadata", "appearance_categories.json"),
    }

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    return {k: v for k, v in kwargs.items() if k in valid_params}

def initialize_attribute_models(face_encoder_root, farl_root):
    # face_encoder_root is the parent dir that contains a `face_encoder/` subdir
    # (matches the layout prepare_face_models expects).
    face_helper_attr, _, _ = prepare_face_models(
        face_encoder_root,
        device=DEVICE,
        dtype=torch.float32,
        clip_opt=False,
    )
    farl_model = prepare_face_attr_model(
        farl_root,
        device=DEVICE,
        dtype=torch.float32,
    )
    return farl_model, face_helper_attr

def convert_to_binary_vector(attr_labels, mapping_list):
    if attr_labels is None:
        return [0] * len(mapping_list)
    attr_set = {a.lower() for a in attr_labels}
    return [1 if m.lower() in attr_set else 0 for m in mapping_list]

def generate_prompt_text(appearance_vec, mapping, categories_dict, allowed_cats):
    prompt_parts = []
    prompt_beginning = ""
    attr_to_cat = {}
    for cat, attrs in categories_dict.items():
        for attr in attrs:
            attr_to_cat[attr] = cat

    def is_allowed(attr_name):
        cat = attr_to_cat.get(attr_name, "non-allowed")
        return cat in allowed_cats

    is_male = False
    try:
        young_idx = mapping.index("young")
        if is_allowed("young"):
            if appearance_vec[young_idx] == 1:
                prompt_beginning += "young "
        male_idx = mapping.index("male")
        if is_allowed("male"):
            is_male = appearance_vec[male_idx] == 1
            prompt_beginning += "male" if is_male else "female"
    except ValueError:
        pass

    for i, val in enumerate(appearance_vec):
        attr_name = mapping[i]
        if attr_name in ("male", "young"):
            continue
        if not is_male and attr_name == "no_beard":
            continue
        if val == 1 and is_allowed(attr_name):
            prompt_parts.append(attr_name.replace("_", " "))

    attrs_str = " with " + ", ".join(prompt_parts) if prompt_parts else ""
    return START_PHRASE + prompt_beginning + attrs_str + END_PHRASE

def build_descriptive_prompt(ref_path, face_attr_model, face_helper_attr, mapping_list, categories_dict, allowed_cats):
    if ref_path is None or not os.path.exists(ref_path):
        print("Reference image not available for descriptive text, falling back to default prompt.")
        return DEFAULT_PROMPT
    ref_bgr = cv2.imread(ref_path)
    if ref_bgr is None:
        print(f"Failed to load reference image {ref_path}, falling back to default prompt.")
        return DEFAULT_PROMPT
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    ref_img_attrs = process_face_attributes(
        face_attr_model,
        face_helper_attr,
        DEVICE,
        torch.float32,
        ref_rgb,
        False,
        PROB_TH,
    )
    binary_img_attrs = convert_to_binary_vector(ref_img_attrs, mapping_list)
    return generate_prompt_text(binary_img_attrs, mapping_list, categories_dict, allowed_cats=allowed_cats)

def initialize_pipeline(paths):
    print("Initializing RGFVRPipeline...")
    config = OmegaConf.load(paths["config"])

    transformer3d = WanRGFVRModel.from_pretrained(
        paths["transformer"],
        config_path=OmegaConf.to_container(config['transformer_kwargs']),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    ).to(DEVICE, dtype=WEIGHT_DTYPE)
    transformer3d = transformer3d.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(paths["model_root"], config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(paths["model_root"], config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=WEIGHT_DTYPE,
    ).eval()

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(paths["model_root"], config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).eval()

    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    face_helper_1, face_helper_2, face_main_model, face_clip_model, _, _ = prepare_face_models(
        paths["face_encoder_root"],
        device=DEVICE,
        dtype=WEIGHT_DTYPE,
        clip_opt=True,
    )

    pipeline = RGFVRPipeline(
        vae=vae.to(WEIGHT_DTYPE),
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        face_helper_1=face_helper_1,
        face_helper_2=face_helper_2,
        face_clip_model=face_clip_model,
        face_main_model=face_main_model,
        transformer=transformer3d,
        scheduler=noise_scheduler,
    ).to(DEVICE)

    return pipeline


def robust_rmtree(path, retries=3):
    if not os.path.exists(path):
        return
    for _ in range(retries):
        try:
            shutil.rmtree(path)
            return
        except OSError:
            time.sleep(0.5)

def process_single_video(video_path, save_dir, clip_name, pipeline, prompt, ref_path,
                         guidance_scale, save_frames=False, fps_override=None):
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Prompt: {prompt}")

    os.makedirs(save_dir, exist_ok=True)
    if save_frames:
        frames_dir = os.path.join(save_dir, "frames")
        robust_rmtree(frames_dir)
        os.makedirs(frames_dir, exist_ok=True)
    else:
        frames_dir = None
    video_save_path = os.path.join(save_dir, f"{clip_name}.mp4")

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        input_fps = vr.get_avg_fps()
        print(f"Video info: {total_frames} frames, {input_fps:.2f} fps")
    except Exception as e:
        print(f"Error reading {video_path}: {e}")
        return

    out_fps = fps_override if fps_override is not None else input_fps
    if fps_override is not None:
        print(f"Output fps overridden to {out_fps}")

    if N_FRAMES is not None:
        limit = min(total_frames, N_FRAMES)
        offset = (limit - 1) % 4
        frames_to_process = limit - offset
    else:
        offset = (total_frames - 1) % 4
        frames_to_process = total_frames - offset

    print(f"Processing {frames_to_process} frames")

    frames_batch = vr.get_batch(range(frames_to_process)).asnumpy()
    frames_pt = torch.Tensor(frames_batch).permute(0, 3, 1, 2)  # [T, C, H, W]

    # The model operates at 512x512. Preserve aspect ratio:
    #   1) resize so the shorter side is 512
    #   2) center-crop to 512x512
    # This avoids the stretching that a direct resize to (512, 512) would cause
    # on non-square videos.
    _, _, H_in, W_in = frames_pt.shape
    if H_in != 512 or W_in != 512:
        short_side = min(H_in, W_in)
        scale = 512.0 / short_side
        new_h = int(round(H_in * scale))
        new_w = int(round(W_in * scale))
        frames_pt = resize(frames_pt, (new_h, new_w))
        top = (new_h - 512) // 2
        left = (new_w - 512) // 2
        frames_pt = frames_pt[:, :, top:top + 512, left:left + 512]
        print(
            f"Input is {W_in}x{H_in} (not 512x512). "
            f"Applied aspect-preserving resize (short-side -> 512, new size {new_w}x{new_h}) "
            f"and center-crop to 512x512."
        )

    frames_pt = frames_pt / 255.0

    control_video = torch.clip(frames_pt, 0.0, 1.0).permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE).to(WEIGHT_DTYPE)

    print("Running inference...")
    try:
        with torch.no_grad():
            with torch.autocast(DEVICE, dtype=WEIGHT_DTYPE):
                generator = torch.Generator(device=DEVICE).manual_seed(SEED)
                output = pipeline(
                    descriptive_prompt=prompt,
                    num_frames=frames_to_process,
                    negative_descriptive_prompt="bad detailed, low quality, distortion",
                    height=512,
                    width=512,
                    generator=generator,
                    guidance_scale=guidance_scale,
                    degraded_video=control_video,
                    reference_image=ref_path,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                ).videos
    except Exception as e:
        print(f"Inference failed: {e}")
        return

    print("Saving results...")
    output_tensor = output[0].permute(1, 2, 3, 0).float().cpu().numpy()
    output_tensor = np.clip(output_tensor * 255, 0, 255).astype(np.uint8)

    T, H, W, _ = output_tensor.shape

    container = av.open(video_save_path, mode='w')
    stream = container.add_stream('libx264', rate=int(round(out_fps)))
    stream.width = W
    stream.height = H
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': '18'}

    for i in range(T):
        frame_rgb = output_tensor[i]
        frame_av = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
        for packet in stream.encode(frame_av):
            container.mux(packet)
        if save_frames:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(frames_dir, f"{i:08d}.png")
            if os.path.exists(save_path):
                os.remove(save_path)
            cv2.imwrite(save_path, frame_bgr)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    print(f"✓ Video saved to: {video_save_path}")
    if save_frames:
        print(f"✓ Frames saved to: {frames_dir}")

    del control_video, frames_pt, frames_batch, output, output_tensor, vr
    gc.collect()
    torch.cuda.empty_cache()


def main():
    args = parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: video_path does not exist: {args.video_path}")
        return

    paths = resolve_ckpt_paths(args.ckpts_dir)
    if not os.path.isdir(paths["ckpts_dir"]):
        print(f"Error: ckpts_dir does not exist: {paths['ckpts_dir']}")
        return
    if not os.path.isfile(paths["config"]):
        print(f"Error: config.yaml not found at {paths['config']}")
        return

    custom_prompt = args.prompt
    effective_degree = 0 if custom_prompt is not None else args.degree

    clip_name = os.path.splitext(os.path.basename(args.video_path))[0]
    save_dir = os.path.join(args.result_path, clip_name)

    print(f"\n{'='*60}")
    print(f"Single Clip Inference: {clip_name}")
    print(f"Ckpts dir: {paths['ckpts_dir']}")
    print(f"Settings: Scale={args.guidance_scale} | Frames={N_FRAMES if N_FRAMES else 'All'} | "
          f"DescriptiveDegree={effective_degree} | CustomPrompt={'yes' if custom_prompt else 'no'}")
    if args.reference_path:
        print(f"Reference image: {args.reference_path}")
    print(f"Output dir: {save_dir}")
    print(f"{'='*60}\n")

    pipeline = initialize_pipeline(paths)

    # Build the prompt
    if custom_prompt is not None:
        prompt = custom_prompt
    elif effective_degree == 0:
        prompt = DEFAULT_PROMPT
    else:
        meta_info = load_json(paths["meta_info"])
        categories = load_json(paths["appearance_categories"])
        mapping_list = meta_info.get("appearance_mapping", [])
        categories_dict = categories["appearance_categories"]
        face_attr_model, face_helper_attr = initialize_attribute_models(
            paths["face_encoder_root"], paths["farl"],
        )
        allowed_cats = DEGREE_TO_CATEGORIES[effective_degree]
        prompt = build_descriptive_prompt(
            args.reference_path, face_attr_model, face_helper_attr,
            mapping_list, categories_dict, allowed_cats,
        )
        del face_attr_model, face_helper_attr
        gc.collect()
        torch.cuda.empty_cache()

    # If the user didn't pass a reference and we don't actually need one,
    # pass None to the pipeline so it doesn't try to open a missing file.
    ref_arg = args.reference_path if (args.reference_path and os.path.exists(args.reference_path)) else None

    process_single_video(
        video_path=args.video_path,
        save_dir=save_dir,
        clip_name=clip_name,
        pipeline=pipeline,
        prompt=prompt,
        ref_path=ref_arg,
        guidance_scale=args.guidance_scale,
        save_frames=args.save_frames,
        fps_override=args.fps,
    )


if __name__ == "__main__":
    main()
