# -*- coding: utf-8 -*-
"""
run_team.py (open-source friendly, compact)

- Single slide:
  --input  one slide patch folder
  --output one .pt

- Batch slides:
  --input root folder with many slide subfolders
  --output_dir output folder
  --batch_slides

- pack-as-slide behavior:
  extract all patch feats in batches -> [N,1024]
  aggregate once -> slide_feat [1,512]
"""

from __future__ import annotations
import os
import argparse
import json
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from team.utils import list_images, list_slide_dirs, save_pt
from team.dataset import PatchImageDataset, collate_fn
from team.patho_team_encoder import EncoderConfig, TEAMPathologyFeatureEncoder
from team.config import load_team_config

WEIGHTS_URL = "https://drive.google.com/drive/folders/1tDbM1GanVYa09wrDsyaqL-F8MmmIlJr5?usp=drive_link"


@torch.no_grad()
def extract_all_patch_feats(encoder, loader, device: str, slide_name: str) -> Tuple[torch.Tensor, List[str]]:
    feats, paths = [], []
    pbar = tqdm(loader, desc=f"Patch batches [{slide_name}]", dynamic_ncols=True)
    for x, ps in pbar:
        x = x.to(device, non_blocking=True)
        out = encoder(x, output_mode="patch", return_aux=False)
        f = out["patch_feat"]  # [B,1024]
        feats.append(f.detach().cpu())
        paths.extend(ps)
        pbar.set_postfix({"total_patches": len(paths)})
    return torch.cat(feats, dim=0), paths  # [N,1024], list


@torch.no_grad()
def aggregate_slide_feat(
    encoder,
    patch_feat_all: torch.Tensor,
    device: str,
    return_aux: bool,
    clinical_text: str = "",
) -> Dict:
    patch_feat_all = patch_feat_all.to(device, non_blocking=True).unsqueeze(0)  # [1,N,1024]
    clinical_feat = None
    if getattr(encoder, "use_text", False) and clinical_text:
        text_vec = encoder.text_encoder([clinical_text], patch_feat_all.device)
        clinical_feat = encoder.text_proj(text_vec)
    slide_feat, aux = encoder.slide_encoder(patch_feat_all, clinical_feat=clinical_feat)  # [1,512]
    out = {"slide_feat": slide_feat.detach().cpu()}
    if return_aux:
        out["aux"] = {k: v.detach().cpu() for k, v in aux.items()}
    return out


def process_one_slide(slide_dir: str, encoder, args, out_path: str, clinical_text: str = "") -> None:
    slide_name = os.path.basename(os.path.normpath(slide_dir))
    img_paths = list_images(slide_dir)
    if len(img_paths) == 0:
        print(f"[WARN] no images found: {slide_dir}")
        return

    ds = PatchImageDataset(img_paths, out_size=args.out_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )

    print(f"\n[INFO] Processing slide: {slide_name} | patches={len(img_paths)}")

    patch_feat_all, paths_all = extract_all_patch_feats(encoder, loader, args.device, slide_name)

    result: Dict = {"paths": paths_all}
    if args.output_mode in ("patch", "both"):
        result["patch_feat"] = patch_feat_all  # [N,1024]

    if args.output_mode in ("slide", "both"):
        result.update(
            aggregate_slide_feat(
                encoder,
                patch_feat_all,
                args.device,
                args.return_aux,
                clinical_text=clinical_text,
            )
        )

    save_pt(result, out_path)

    if "slide_feat" in result:
        print(f"[OK] Saved: {out_path} | slide_feat={tuple(result['slide_feat'].shape)}")
    else:
        print(f"[OK] Saved: {out_path}")


def main():
    p = argparse.ArgumentParser("run_team")
    p.add_argument("--config", type=str, default="configs/team_config.json",
                   help="path to TEAM config JSON")
    p.add_argument("--clinical_text", type=str, default=None,
                   help="single-slide clinical free-text (used only when upstream.model.use_text=true)")
    p.add_argument("--text_json", type=str, default=None,
                   help="json file mapping slide_name -> clinical text for batch mode")

    p.add_argument("--input", type=str, default=None,
                   help="single-slide patch folder; or (batch) root folder with many slide subfolders")
    p.add_argument("--output", type=str, default=None,
                   help="single-slide output .pt path (unused in batch mode)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="batch mode output dir (one .pt per slide)")

    p.add_argument("--output_mode", type=str, default=None, choices=["patch", "slide", "both"])
    p.add_argument("--return_aux", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--out_size", type=int, default=None)
    p.add_argument("--use_fp16", action="store_true")
    p.add_argument("--use_text", action="store_true")

    p.add_argument("--batch_slides", action="store_true",
                   help="batch mode: each first-level subfolder under --input is treated as one slide")
    p.add_argument("--limit_slides", type=int, default=None)

    # checkpoints: provided by CLI (no real paths)
    p.add_argument("--patch_ckpt", type=str, default=None)
    p.add_argument("--slide_ckpt", type=str, default=None)
    p.add_argument("--enc_name", type=str, default=None)

    args = p.parse_args()

    cfg_all = load_team_config(args.config)
    up_model = cfg_all["upstream"]["model"]
    up_runtime = cfg_all["upstream"]["runtime"]
    up_paths = cfg_all["upstream"]["paths"]

    args.input = args.input if args.input is not None else up_paths.get("input", "")
    args.output = args.output if args.output is not None else up_paths.get("output", "")
    args.output_dir = args.output_dir if args.output_dir is not None else up_paths.get("output_dir", "")
    args.clinical_text = args.clinical_text if args.clinical_text is not None else up_paths.get("clinical_text", "")
    args.text_json = args.text_json if args.text_json is not None else up_paths.get("text_json", "")
    args.patch_ckpt = args.patch_ckpt if args.patch_ckpt is not None else up_paths.get("patch_ckpt", "")
    args.slide_ckpt = args.slide_ckpt if args.slide_ckpt is not None else up_paths.get("slide_ckpt", "")

    args.output_mode = args.output_mode if args.output_mode is not None else up_runtime.get("output_mode", "slide")
    args.device = args.device if args.device is not None else up_runtime.get("device", "cuda")
    args.batch_size = args.batch_size if args.batch_size is not None else int(up_runtime.get("batch_size", 128))
    args.num_workers = args.num_workers if args.num_workers is not None else int(up_runtime.get("num_workers", 4))
    args.out_size = args.out_size if args.out_size is not None else int(up_runtime.get("out_size", 224))
    args.limit_slides = args.limit_slides if args.limit_slides is not None else int(up_paths.get("limit_slides", -1))
    args.enc_name = args.enc_name if args.enc_name is not None else up_model.get("patch_enc_name")
    if not args.batch_slides:
        args.batch_slides = bool(up_paths.get("batch_slides", False))
    if not args.return_aux:
        args.return_aux = bool(up_runtime.get("return_aux", False))
    if not args.use_fp16:
        args.use_fp16 = bool(up_runtime.get("use_fp16", False))
    if not args.use_text:
        args.use_text = bool(up_model.get("use_text", False))
    if not args.use_text and (bool(args.clinical_text) or bool(args.text_json)):
        args.use_text = True

    if not args.input:
        raise ValueError("input is required: set --input or upstream.paths.input in config")
    if not args.patch_ckpt:
        raise ValueError(
            "patch_ckpt is required: set --patch_ckpt or upstream.paths.patch_ckpt in config. "
            f"Official weights: {WEIGHTS_URL}"
        )
    if not args.slide_ckpt:
        raise ValueError(
            "slide_ckpt is required: set --slide_ckpt or upstream.paths.slide_ckpt in config. "
            f"Official weights: {WEIGHTS_URL}"
        )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA is not available. Fallback to CPU.")
        args.device = "cpu"

    cfg = EncoderConfig(
        patch_enc_name=args.enc_name,
        patch_ckpt=args.patch_ckpt,
        slide_ckpt=args.slide_ckpt,
        device=args.device,
        use_fp16=args.use_fp16,
        patch_feat_dim=int(up_model.get("patch_feat_dim", 1024)),
        slide_feat_dim=int(up_model.get("slide_feat_dim", 512)),
        uncertainty_samples=int(up_model.get("uncertainty_samples", 3)),
        dropout_rate=float(up_model.get("dropout_rate", 0.3)),
        attn_hidden_dim=int(up_model.get("attn_hidden_dim", 256)),
        backbone_model_name=str(up_model.get("backbone_model_name", "vit_large_patch16_224")),
        backbone_img_size=int(up_model.get("backbone_img_size", 224)),
        backbone_patch_size=int(up_model.get("backbone_patch_size", 16)),
        backbone_init_values=float(up_model.get("backbone_init_values", 1e-5)),
        use_text=bool(args.use_text),
        text_model_name=str(up_model.get("text_model_name", "dmis-lab/biobert-base-cased-v1.2")),
        text_max_length=int(up_model.get("text_max_length", 256)),
    )
    encoder = TEAMPathologyFeatureEncoder(cfg).eval()

    slide_text_map = {}
    if args.text_json:
        with open(args.text_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"text_json must be a JSON object: {args.text_json}")
        slide_text_map = {str(k): str(v) for k, v in data.items()}

    if args.batch_slides:
        if args.output_dir == "":
            raise ValueError("batch mode requires --output_dir")
        slide_dirs = list_slide_dirs(args.input)
        if args.limit_slides > 0:
            slide_dirs = slide_dirs[:args.limit_slides]

        slides_pbar = tqdm(slide_dirs, desc="Slides", dynamic_ncols=True)
        for sd in slides_pbar:
            name = os.path.basename(os.path.normpath(sd))
            slides_pbar.set_description(f"Processing slide: {name}")
            out_path = os.path.join(args.output_dir, f"{name}.pt")
            process_one_slide(sd, encoder, args, out_path, clinical_text=slide_text_map.get(name, ""))
        print(f"\n[ALL DONE] output_dir: {args.output_dir}")
        return

    if args.output == "":
        raise ValueError("single-slide mode requires --output")
    process_one_slide(args.input, encoder, args, args.output, clinical_text=args.clinical_text)


if __name__ == "__main__":
    main()
