"""
Face embedding and attribute utilities for RFVR restoration pipeline.

Adapted from the ConsisID project:
    https://github.com/PKU-YuanGroup/ConsisID
"""

import os
import math
from typing import List, Optional, Tuple, Union

import cv2
import insightface
import numpy as np
import torch

from eva_clip import create_model_and_transforms
from eva_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from insightface.app import FaceAnalysis
from PIL import Image, ImageOps
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

from . import iresnet
from .farl import FaRLFaceAttribute
from .model_irse import Backbone as IRBackbone


def resize_numpy_image_long(image, resize_long_edge=768):
    h, w = image.shape[:2]
    if max(h, w) <= resize_long_edge:
        return image
    k = resize_long_edge / max(h, w)
    h = int(h * k)
    w = int(w * k)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def img2tensor(imgs, bgr2rgb=True, float32=True):
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == "float64":
                img = img.astype("float32")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1).copy())
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    return _totensor(imgs, bgr2rgb, float32)


def to_gray(img):
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    x = x.repeat(1, 3, 1, 1)
    return x


def prepare_face_attr_model(model_path, device, dtype):
    model_name = "face_attribute.farl.celeba.pt"
    conf_name = "celeba/224"
    model = FaRLFaceAttribute(conf_name=conf_name, model_path=os.path.join(model_path, model_name))
    model.eval()
    model.to(device, dtype=dtype)
    return model


def calculate_kps(image, face_helper_1):
    face_helper_1.clean_all()
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    face_helper_1.read_image(image_bgr)
    face_helper_1.get_face_landmarks_5(only_keep_largest=True)
    face_kps = face_helper_1.all_landmarks_5[0]
    return face_kps


def process_face_attributes(
    face_attr_model,
    face_helper_1,
    device,
    dtype,
    image,
    return_probs: bool = True,
    prob_th: float = 0.5,
):
    face_kps = None
    if face_helper_1 is not None:
        face_kps = calculate_kps(image, face_helper_1)
    input = img2tensor(image, False).unsqueeze(0)
    input = input.to(device, dtype=dtype)
    kps = torch.Tensor(face_kps).unsqueeze(0).to(device, dtype=dtype)

    with torch.no_grad():
        probs = face_attr_model(input, kps)

    probs = probs.squeeze(0).detach().cpu().tolist()
    if return_probs:
        return probs
    attrs = []
    labels = face_attr_model.labels
    for attr, prob in zip(labels, probs):
        if prob > prob_th:
            attrs.append(attr)
    return attrs


def create_fr_model(model_path, depth="100", use_amp=False):
    model = iresnet(depth)
    model.load_state_dict(torch.load(model_path))
    if use_amp:
        model.half()
    return model


def prepare_curricular_face_model(model_path, device, dtype):
    model = IRBackbone([112, 112], num_layers=100, mode='ir')
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    model.to(device, dtype=dtype)
    return model


def prepare_face_models(model_path, device, dtype, clip_opt: bool = False):
    """
    Prepare face models for the restoration pipeline.

    Returns:
        If clip_opt=False: (face_helper_1, face_helper_2, face_main_model)
        If clip_opt=True:  (face_helper_1, face_helper_2, face_main_model,
                            face_clip_model, eva_transform_mean, eva_transform_std)
    """
    face_helper_1 = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        device=device,
        model_rootpath=os.path.join(model_path, "face_encoder"),
    )
    face_helper_1.face_parse = None
    face_helper_1.face_parse = init_parsing_model(
        model_name="bisenet", device=device, model_rootpath=os.path.join(model_path, "face_encoder")
    )

    face_helper_2 = insightface.model_zoo.get_model(
        f"{model_path}/face_encoder/models/antelopev2/glintr100.onnx", providers=["CUDAExecutionProvider"]
    )
    face_helper_2.prepare(ctx_id=0)

    face_main_model = FaceAnalysis(
        name="antelopev2", root=os.path.join(model_path, "face_encoder"), providers=["CUDAExecutionProvider"]
    )
    face_main_model.prepare(ctx_id=0, det_size=(640, 640))

    face_helper_1.face_det.eval()
    face_helper_1.face_parse.eval()
    face_helper_1.face_det.to(device)
    face_helper_1.face_parse.to(device)

    if clip_opt:
        model, _, _ = create_model_and_transforms(
            "EVA02-CLIP-L-14-336",
            os.path.join(model_path, "face_encoder", "EVA02_CLIP_L_336_psz14_s6B.pt"),
            force_custom_clip=True,
        )
        face_clip_model = model.visual
        eva_transform_mean = getattr(face_clip_model, "image_mean", OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(face_clip_model, "image_std", OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            eva_transform_mean = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            eva_transform_std = (eva_transform_std,) * 3
        face_clip_model.eval()
        face_clip_model.to(device, dtype=dtype)
        face_clip_model.dtype = dtype
        face_clip_model.device = device

        return face_helper_1, face_helper_2, face_main_model, face_clip_model, eva_transform_mean, eva_transform_std

    return face_helper_1, face_helper_2, face_main_model


def process_face_embeddings(
    app,
    face_helper_1,
    face_helper_2,
    device,
    weight_dtype,
    image,
    clip_vision_model,
    eva_transform_mean,
    eva_transform_std,
    gray_opt: bool = False,
    return_aligned_image: bool = False,
):
    """
    Extract face embeddings (ArcFace + EVA-CLIP) from a single reference image.

    Returns:
        id_cond:        (1, 1280) concatenated ArcFace + CLIP global embedding
        id_vit_hidden:  (1, 577, 1024) CLIP hidden states
        [return_face_features_image]: (1, 3, 512, 512) aligned face (if return_aligned_image)
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    id_ante_embedding = None
    face_kps = None
    align_face = None
    return_face_features_image = None

    face_info = app.get(image_bgr)
    if len(face_info) > 0:
        face_info = sorted(face_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1]
        id_ante_embedding = face_info["embedding"]
        face_kps = face_info["kps"]

    try:
        face_helper_1.clean_all()
        face_helper_1.read_image(image_bgr)
        face_helper_1.get_face_landmarks_5(only_keep_largest=True)
        face_helper_1.align_warp_face()

        if len(face_helper_1.cropped_faces) > 0:
            align_face = face_helper_1.cropped_faces[0]
            if id_ante_embedding is None:
                id_ante_embedding = face_helper_2.get_feat(align_face)
            if face_kps is None and len(face_helper_1.all_landmarks_5) > 0:
                face_kps = face_helper_1.all_landmarks_5[0]
    except Exception as e:
        print(f"Warning: Face Alignment/Fallback failed: {e}")

    if id_ante_embedding is not None:
        id_ante_embedding = torch.from_numpy(id_ante_embedding).to(device, weight_dtype)
        if id_ante_embedding.ndim == 1:
            id_ante_embedding = id_ante_embedding.unsqueeze(0)

        if align_face is not None:
            input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
            input = input.to(device)

            parsing_out = face_helper_1.face_parse(
                normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            )[0]
            parsing_out = parsing_out.argmax(dim=1, keepdim=True)
            bg_label = [0, 16, 18, 9, 14, 15]
            bg = sum(parsing_out == i for i in bg_label).bool()
            white_image = torch.ones_like(input)

            if gray_opt:
                return_face_features_image = torch.where(bg, white_image, to_gray(input))
            else:
                return_face_features_image = torch.where(bg, white_image, input)

            face_features_image = resize(
                return_face_features_image, clip_vision_model.image_size, InterpolationMode.BICUBIC
            )
            face_features_image = normalize(face_features_image, eva_transform_mean, eva_transform_std)

            global_clip_embedding, local_clip_embeddings = clip_vision_model(
                face_features_image.to(weight_dtype), return_all_features=False, return_hidden=True, shuffle=False
            )

            global_clip_embedding_norm = torch.norm(global_clip_embedding, 2, 1, True)
            global_clip_embedding = torch.div(global_clip_embedding, global_clip_embedding_norm)

            global_perceptual_embedding = torch.cat([id_ante_embedding, global_clip_embedding], dim=-1)
            local_perceptual_embeddings = local_clip_embeddings[-1] # late-stage patch-level features
        else:
            global_perceptual_embedding = torch.zeros((1, 1280), dtype=weight_dtype, device=device)
            local_perceptual_embeddings = torch.zeros((1, 577, 1024), dtype=weight_dtype, device=device)
    else:
        global_perceptual_embedding = torch.zeros((1, 1280), dtype=weight_dtype, device=device)
        local_perceptual_embeddings = torch.zeros((1, 577, 1024), dtype=weight_dtype, device=device)

    if return_aligned_image:
        return global_perceptual_embedding, local_perceptual_embeddings, return_face_features_image
    return global_perceptual_embedding, local_perceptual_embeddings

def parse_image(image, face_helper_1, device, return_pt: bool = False, is_warp: bool = False):
    if is_warp:
        image = warp_image(image, face_helper_1)
    image = image[:, :, ::-1]
    input = img2tensor(image, bgr2rgb=True).unsqueeze(0) / 255.0
    input = input.to(device)
    parsing_out = face_helper_1.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
    parsing_out = parsing_out.argmax(dim=1, keepdim=True)
    bg_label = [0, 16, 18, 9, 14, 15]
    bg = sum(parsing_out == i for i in bg_label).bool()
    white_image = torch.ones_like(input)

    parsed_image = torch.where(bg, white_image, input)
    parsed_image = (parsed_image * 255.0).type(torch.uint8)
    parsed_image = parsed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if return_pt:
        return parsed_image
    return parsed_image

def warp_image(image, face_helper_1):
    face_helper_1.clean_all()
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    face_helper_1.read_image(image_bgr)
    face_helper_1.get_face_landmarks_5(only_keep_largest=True)
    face_kps = face_helper_1.all_landmarks_5[0]
    face_helper_1.align_warp_face()
    if len(face_helper_1.cropped_faces) == 0:
        raise RuntimeError("facexlib align face fail")
    align_face = face_helper_1.cropped_faces[0]
    align_face = align_face[:, :, ::-1]
    return align_face

def return_kps(image, face_helper_1):
    face_helper_1.clean_all()
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    face_helper_1.read_image(image_bgr)
    face_helper_1.get_face_landmarks_5(only_keep_largest=True)
    face_kps = face_helper_1.all_landmarks_5[0]
    return face_kps


def tensor_to_pil(src_img_tensor):
    img = src_img_tensor.clone().detach()
    if img.dtype == torch.bfloat16:
        img = img.to(torch.float32)
    img = img.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.uint8)
    return Image.fromarray(img)
