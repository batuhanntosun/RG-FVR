# Subject-Agnostic Identity-Preserving Face Video Restoration with a Reference Guidance

Single-clip face video restoration (**RGFVR**) using a high-quality reference image of the target subject.

---

## Method overview

Given a degraded face video and a high-quality reference image of the same subject, RGFVR restores the video while preserving the subject's identity. It conditions the diffusion model on two complementary feature types extracted from the reference image.

**Perceptual features** — *who* the person is:
- **ArcFace** (iResNet / AntelopeV2, InsightFace) — global identity embedding
- **EVA02-CLIP-L-14-336** (EVA-ViT) — global CLIP embedding + local patch-level features

**Descriptive features** — *how* the person looks:
- **FaRL** (Face Representation Learning)\* predicts face attributes from the reference image
- Attributes are filtered by a confidence threshold and a selectable detail level (`--degree`)
- The resulting description is encoded by the **WanT5** text encoder into conditioning embeddings

**Input requirements:**
- Video: 512×512 resolution. Non-square inputs are resized (short-side → 512) and center-cropped automatically.
- Optimal clip length: **81 frames**. Longer clips may work but are not fully validated.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (tested on CUDA 12.8)

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv rgfvr_env
source rgfvr_env/bin/activate

# 2. Install PyTorch with the appropriate CUDA build (see https://pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Install remaining dependencies
pip install -r requirements.txt
```

## Checkpoints

Place all model weights under a single directory (default: `ckpts/`). Expected layout:

```
ckpts/
├── config.yaml
├── transformer/          # WanRGFVRModel weights (teacher)
├── dmd/
│   └── generator_lora.safetensors  # DMD LoRA weights (3-step distilled)
├── vae/
├── text_encoder/         # WanT5
├── tokenizer/
├── scheduler/
├── face_encoder/         # ArcFace (AntelopeV2) + EVA02-CLIP-L-14-336
├── farl/                 # FaRL attribute predictor
├── arcface/
└── metadata/
    ├── meta_info.json
    └── appearance_categories.json
```

The `dmd/` subfolder is only required for DMD inference (`inference_dmd.py`).

---

## Usage

Two inference modes are available. Both share the same checkpoint directory and produce identical output formats.

### Standard (50-step)

Full-quality 50-step flow matching with classifier-free guidance.

```bash
python inference.py \
    --video_path     path/to/degraded.mp4 \
    --reference_path path/to/reference.jpg \
    --result_path    path/to/output_dir
```

### DMD (3-step, fast)

3-step DMD-distilled inference. Significantly faster with comparable quality. No classifier-free guidance.

```bash
python inference_dmd.py \
    --video_path     path/to/degraded.mp4 \
    --reference_path path/to/reference.jpg \
    --result_path    path/to/output_dir
```

The output video is written to `<result_path>/<clip_name>/<clip_name>.mp4` in both cases.

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--video_path` | `./video.mp4` | Degraded input video (512×512 recommended) |
| `--reference_path` | `./ref.jpg` | High-quality reference image of the target subject |
| `--result_path` | `.` | Output directory |
| `--degree` | `4` | Descriptive prompt detail level (see below) |
| `--prompt` | — | Custom text prompt; overrides `--degree` |
| `--guidance_scale` | `2.0` | CFG scale — **standard only**, ignored by DMD |
| `--fps` | input fps | Output video FPS |
| `--save_frames` | off | Also save each output frame as PNG |
| `--ckpts_dir` | `./ckpts` | Checkpoint directory |

### `--degree`

Controls the level of face-attribute detail included in the descriptive conditioning prompt. Attributes are predicted automatically from the reference image via FaRL.

| Degree | Attributes |
|---|---|
| `0` | Generic prompt (no attribute prediction) |
| `1` | Gender |
| `2` | Gender + hair |
| `3` | Gender + hair + facial hair |
| `4` | All attributes: gender + hair + facial hair + accessories / makeup *(default)* |

Use a lower degree if the reference image is loosely aligned with the subject in the degraded video.

### Examples

```bash
# Standard — full attribute conditioning
python inference.py --video_path video.mp4 --reference_path ref.jpg --result_path out/

# DMD — same interface, ~17× fewer denoising steps
python inference_dmd.py --video_path video.mp4 --reference_path ref.jpg --result_path out/

# More robust when reference is loosely aligned (works for both modes)
python inference.py --video_path video.mp4 --reference_path ref.jpg --result_path out/ --degree 1

# Custom prompt
python inference.py --video_path video.mp4 --reference_path ref.jpg --result_path out/ \
    --prompt "A photorealistic close-up video of a young woman, 4k, high temporal consistency."

# Save individual frames alongside the video
python inference_dmd.py --video_path video.mp4 --reference_path ref.jpg --result_path out/ --save_frames
```
