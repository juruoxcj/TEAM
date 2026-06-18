# -*- coding: utf-8 -*-
"""Minimal no-data smoke test for the downstream TEAM model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from team.biomarker_driven_team import BiomarkerDrivenTEAMModel
from team.config import load_team_config


def main() -> int:
    parser = argparse.ArgumentParser("TEAM smoke test")
    parser.add_argument("--config", type=str, default="configs/team_config.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_patches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA is not available. Fallback to CPU.")
        args.device = "cpu"

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg = load_team_config(args.config)
    d_model = cfg["downstream"]["model"]
    patch_feat_dim = int(d_model.get("patch_feat_dim", 1024))
    num_cancers = int(d_model.get("num_cancers", 32))

    model = BiomarkerDrivenTEAMModel.from_config(
        args.config,
        device=device,
        overrides={
            "use_stage": True,
            "use_tme": False,
            "use_gene": False,
            "slide_ckpt": "",
            "stage_ckpt": "",
            "tme_ckpt": "",
            "gene_ckpt": "",
        },
    ).to(device).eval()

    feat_tensor = torch.randn(1, args.num_patches, patch_feat_dim, device=device)
    cancer_id = torch.zeros(1, dtype=torch.long, device=device)
    if num_cancers < 1:
        raise ValueError("downstream.model.num_cancers must be >= 1")

    with torch.no_grad():
        pred, fused_feat = model(feat_tensor, cancer_id)

    print(f"pred shape: {tuple(pred.shape)}")
    print(f"fused_feat shape: {tuple(fused_feat.shape)}")
    print("[OK] TEAM downstream smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
