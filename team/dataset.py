# -*- coding: utf-8 -*-
"""
team_dataset.py
Patch image dataset + preprocessing (resize & ImageNet norm).
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def preprocess_pil(
    img: Image.Image,
    out_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((out_size, out_size), resample=Image.BICUBIC)

    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # [C,H,W]
    x = torch.from_numpy(arr)

    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t  = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (x - mean_t) / std_t


class PatchImageDataset(Dataset):
    def __init__(self, image_paths: List[str], out_size: int = 224):
        self.image_paths = image_paths
        self.out_size = out_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        p = self.image_paths[idx]
        img = Image.open(p)
        x = preprocess_pil(img, out_size=self.out_size)
        return x, p


def collate_fn(batch):
    xs, ps = zip(*batch)
    x = torch.stack(xs, dim=0)  # [B,3,224,224]
    return x, list(ps)
