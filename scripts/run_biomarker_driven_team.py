# -*- coding: utf-8 -*-
"""Run BiomarkerDrivenTEAMModel on extracted TEAM patch features."""

from __future__ import annotations
import argparse

import torch

from team.biomarker_driven_team import BiomarkerDrivenTEAMModel
from team.config import load_team_config


def main():
    parser = argparse.ArgumentParser("run_biomarker_driven_team")
    parser.add_argument("--config", type=str, default="configs/team_config.json")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_patches", type=int, default=None)
    parser.add_argument("--num_cancers", type=int, default=None)
    parser.add_argument("--input_pt", type=str, required=True,
                        help="path to one upstream .pt file containing patch_feat")
    parser.add_argument("--use_stage", action="store_true")
    parser.add_argument("--use_tme", action="store_true")
    parser.add_argument("--use_gene", action="store_true")
    args = parser.parse_args()

    cfg = load_team_config(args.config)
    d_runtime = cfg["downstream"]["runtime"]
    d_model = cfg["downstream"]["model"]
    args.device = args.device if args.device is not None else d_runtime.get("device", "cuda")
    args.batch_size = args.batch_size if args.batch_size is not None else int(d_runtime.get("batch_size", 2))
    args.num_patches = args.num_patches if args.num_patches is not None else int(d_runtime.get("num_patches", 256))
    args.num_cancers = args.num_cancers if args.num_cancers is not None else int(d_model.get("num_cancers", 32))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA is not available. Fallback to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    model = BiomarkerDrivenTEAMModel.from_config(
        args.config,
        device=device,
        overrides={
            "use_stage": bool(d_model.get("use_stage", True)) or args.use_stage,
            "use_tme": bool(d_model.get("use_tme", False)) or args.use_tme,
            "use_gene": bool(d_model.get("use_gene", False)) or args.use_gene,
            "num_cancers": args.num_cancers,
        },
    ).to(device).eval()

    with torch.no_grad():
        pack = torch.load(args.input_pt, map_location="cpu")
        if "patch_feat" not in pack:
            raise KeyError(f"patch_feat not found in {args.input_pt}; rerun upstream with --output_mode both/patch")
        feat_tensor = pack["patch_feat"]
        feat_tensor = feat_tensor.unsqueeze(0).to(device)
        bsz = feat_tensor.size(0)
        cancer_id = torch.randint(0, args.num_cancers, (bsz,), device=device, dtype=torch.long)
        pred, fused_feat = model(feat_tensor, cancer_id)

    print(f"pred shape:        {tuple(pred.shape)}")
    print(f"fused_feat shape:   {tuple(fused_feat.shape)}")


if __name__ == "__main__":
    main()
