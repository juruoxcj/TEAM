# -*- coding: utf-8 -*-
"""
team_utils.py
Open-source friendly utils (no hard-coded real paths).
"""

from __future__ import annotations
import os
import glob
from typing import List, Dict, Any

import torch

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def list_images(folder_or_file: str) -> List[str]:
    """List image files under a folder (or return [file] if a file path is given)."""
    if os.path.isfile(folder_or_file):
        return [folder_or_file]

    paths: List[str] = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder_or_file, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder_or_file, f"*{ext.upper()}")))
    return sorted(list(set(paths)))


def list_slide_dirs(root_dir: str) -> List[str]:
    """Treat each first-level subfolder under root_dir as one slide folder."""
    slide_dirs: List[str] = []
    for name in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p):
            slide_dirs.append(p)
    return slide_dirs


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if d:
        os.makedirs(d, exist_ok=True)


def save_pt(obj: Dict[str, Any], out_path: str) -> None:
    ensure_dir(out_path)
    torch.save(obj, out_path)
