# -*- coding: utf-8 -*-
"""
Centralized configuration loader for TEAM.
"""

from __future__ import annotations
import copy
import json
import os
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "upstream": {
        "model": {
            "patch_enc_name": "vit_large_patch16_224.dinov2.uni_mass100k",
            "backbone_model_name": "vit_large_patch16_224",
            "backbone_img_size": 224,
            "backbone_patch_size": 16,
            "backbone_init_values": 1e-5,
            "patch_feat_dim": 1024,
            "slide_feat_dim": 512,
            "uncertainty_samples": 3,
            "dropout_rate": 0.3,
            "attn_hidden_dim": 256,
            "use_text": False,
            "text_model_name": "dmis-lab/biobert-base-cased-v1.2",
            "text_max_length": 256
        },
        "runtime": {
            "device": "cuda",
            "batch_size": 128,
            "num_workers": 4,
            "out_size": 224,
            "use_fp16": False,
            "output_mode": "slide",
            "return_aux": False,
        },
        "paths": {
            "patch_ckpt": "./patch_weight.pth",
            "slide_ckpt": "./slide_weight.pth",
            "input": "",
            "output": "",
            "output_dir": "",
            "clinical_text": "",
            "text_json": "",
            "batch_slides": False,
            "limit_slides": -1,
        },
    },
    "downstream": {
        "model": {
            "patch_feat_dim": 1024,
            "shared_dim": 512,
            "stage_classes": 4,
            "tme_classes": 8,
            "gene_classes": 15717,
            "num_cancers": 32,
            "tau": 1.1,
            "use_stage": True,
            "use_tme": False,
            "use_gene": False,
        },
        "paths": {
            "slide_ckpt": "",
            "stage_ckpt": "",
            "tme_ckpt": "",
            "gene_ckpt": "",
        },
        "runtime": {
            "device": "cuda",
            "batch_size": 1,
            "num_patches": 256,
        },
    },
}


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_team_config(config_path: str = "configs/team_config.json") -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if not config_path:
        return cfg
    if not os.path.exists(config_path):
        return cfg
    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    if not isinstance(user_cfg, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    return _deep_update(cfg, user_cfg)
