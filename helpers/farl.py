import os
import functools
from typing import Optional

import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from facer import FaceAttribute
from facer.farl import farl_classification
from facer.transform import get_face_align_matrix, make_tanh_warp_grid

def get_std_points_xray(out_size=256, mid_size=500):
    std_points_256 = np.array(
        [
            [85.82991, 85.7792],
            [169.0532, 84.3381],
            [127.574, 137.0006],
            [90.6964, 174.7014],
            [167.3069, 173.3733],
        ]
    )
    std_points_256[:, 1] += 30
    old_size = 256
    mid = mid_size / 2
    new_std_points = std_points_256 - old_size / 2 + mid
    target_pts = new_std_points * out_size / mid_size
    target_pts = torch.from_numpy(target_pts).float()
    return target_pts

pretrain_settings = {
    "celeba/224": {
        "num_classes": 40,
        "layers": [11],
        "url": "https://github.com/FacePerceiver/facer/releases/download/models-v1/face_attribute.farl.celeba.pt",
        "get_matrix_fn": functools.partial(
            get_face_align_matrix,
            target_shape=(224, 224),
            target_pts=get_std_points_xray(out_size=224, mid_size=500),
        ),
        "get_grid_fn": functools.partial(
            make_tanh_warp_grid, warp_factor=0.0, warped_shape=(224, 224)
        ),
        "classes": [
            "5_o_Clock_Shadow",
            "Arched_Eyebrows",
            "Attractive",
            "Bags_Under_Eyes",
            "Bald",
            "Bangs",
            "Big_Lips",
            "Big_Nose",
            "Black_Hair",
            "Blond_Hair",
            "Blurry",
            "Brown_Hair",
            "Bushy_Eyebrows",
            "Chubby",
            "Double_Chin",
            "Eyeglasses",
            "Goatee",
            "Gray_Hair",
            "Heavy_Makeup",
            "High_Cheekbones",
            "Male",
            "Mouth_Slightly_Open",
            "Mustache",
            "Narrow_Eyes",
            "No_Beard",
            "Oval_Face",
            "Pale_Skin",
            "Pointy_Nose",
            "Receding_Hairline",
            "Rosy_Cheeks",
            "Sideburns",
            "Smiling",
            "Straight_Hair",
            "Wavy_Hair",
            "Wearing_Earrings",
            "Wearing_Hat",
            "Wearing_Lipstick",
            "Wearing_Necklace",
            "Wearing_Necktie",
            "Young",
        ],
    }
}

def load_face_attr(model_path, num_classes=40, layers=[11]):
    model = farl_classification(num_classes=num_classes, layers=layers)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model

class FaRLFaceAttribute(FaceAttribute):
    def __init__(
        self,
        conf_name: Optional[str] = None,
        model_path: Optional[str] = None,
        device=None,
    ) -> None:
        super().__init__()
        if conf_name is None:
            conf_name = "celeba/224"
        if model_path is None:
            model_path = pretrain_settings[conf_name]["url"]
        self.conf_name = conf_name
        self.setting = pretrain_settings[self.conf_name]
        self.labels = self.setting["classes"]
        self.net = load_face_attr(model_path, 
                                  num_classes=self.setting["num_classes"], 
                                  layers = self.setting["layers"])
        if device is not None:
            self.net = self.net.to(device)
        self.normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.eval()

    def forward(self, images: torch.Tensor, kps: Optional[torch.Tensor]):
        images = images.float() / 255.0
        _, _, h, w = images.shape
        if kps is not None:
            matrix = self.setting["get_matrix_fn"](kps)
            grid = self.setting["get_grid_fn"](matrix=matrix, orig_shape=(h, w))
            w_images = F.grid_sample(images, 
                                    grid, 
                                    mode="bilinear", 
                                    align_corners=False)
        else:
            img_preprocess = transforms.Compose(
                                    [
                                        transforms.Resize((224,224)),
                                        transforms.CenterCrop((224, 224)),
                                    ]
                            )
            w_images = img_preprocess(images)
        w_images = self.normalize(w_images)
        outputs = self.net(w_images)
        probs = torch.sigmoid(outputs)
        return probs