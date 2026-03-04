# -*- coding: utf-8 -*-
"""Core encoders for TEAM upstream feature extraction."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union, Literal

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlideEncoderFeatOnly(nn.Module):
    def __init__(
        self,
        patch_feat_dim: int = 1024,
        slide_feat_dim: int = 512,
        uncertainty_samples: int = 3,
        dropout_rate: float = 0.3,
        attn_hidden_dim: int = 256,
        finetune: bool = False,
    ):
        super().__init__()
        if uncertainty_samples < 1:
            raise ValueError("uncertainty_samples must be >= 1")
        self.patch_feat_dim = patch_feat_dim
        self.slide_feat_dim = slide_feat_dim
        self.uncertainty_samples = uncertainty_samples
        self.dropout_rate = dropout_rate

        self.modules_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.patch_feat_dim, self.slide_feat_dim),
                nn.LayerNorm(self.slide_feat_dim),
                nn.ReLU(),
                nn.Linear(self.slide_feat_dim, self.slide_feat_dim),
                nn.LayerNorm(self.slide_feat_dim),
                nn.ReLU()
            ) for _ in range(2 * self.uncertainty_samples)
        ])
        self.attn = nn.Sequential(
            nn.Linear(self.slide_feat_dim, self.slide_feat_dim),
            nn.LayerNorm(self.slide_feat_dim),
            nn.ReLU(),
            nn.Linear(self.slide_feat_dim, attn_hidden_dim),
            nn.LayerNorm(attn_hidden_dim),
            nn.ReLU(),
            nn.Linear(attn_hidden_dim, 1),
            nn.ReLU()
        )
        self.attn_clinical = nn.Linear(self.slide_feat_dim, 1)

        if finetune:
            for p in self.modules_list.parameters():
                p.requires_grad = False
            for p in self.attn.parameters():
                p.requires_grad = False

    @staticmethod
    def _minmax_norm(x: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
        x_min = x.min(dim=dim, keepdim=True)[0]
        x_max = x.max(dim=dim, keepdim=True)[0]
        return (x - x_min) / (x_max - x_min + eps)

    def uncertainty_model(
        self,
        inputs: torch.Tensor,
        clinical_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feat_mean1, feat_var_log = [], []
        feat = inputs.float()

        for i in range(self.uncertainty_samples):
            feat_drop = F.dropout(feat, p=self.dropout_rate, training=True)
            idx = i * 2
            feat_mean1.append(self.modules_list[idx](feat_drop))
            feat_var_log.append(self.modules_list[idx + 1](feat_drop))

        output_mean1 = torch.stack(feat_mean1, dim=1)
        output_var = torch.stack(feat_var_log, dim=1).exp()

        zp = torch.mean(output_mean1, dim=1)
        var1 = torch.var(output_mean1, dim=1)
        var1 = torch.mean(var1, dim=2)

        u1_raw = torch.mean(output_var, dim=(1, 3))
        u1 = self._minmax_norm(u1_raw, dim=1)

        u2 = torch.sigmoid(10 * (var1 - 0.5))
        u2 = self._minmax_norm(u2, dim=1)

        attn_weight_patch = self.attn(zp)
        if attn_weight_patch.dim() == 3:
            attn_weight_patch = attn_weight_patch.squeeze(2)
        attn_weight_patch = self._minmax_norm(attn_weight_patch, dim=1)

        weight_logits_patch = attn_weight_patch * (1 - u1) * (1 - u2)

        if clinical_feat is not None:
            if clinical_feat.dim() != 2 or clinical_feat.size(1) != self.slide_feat_dim:
                raise ValueError(
                    f"clinical_feat should be [B,{self.slide_feat_dim}], got {tuple(clinical_feat.shape)}"
                )
            clinical_logit = self.attn_clinical(clinical_feat)
            logits_all = torch.cat([clinical_logit, weight_logits_patch], dim=1)
            weight_all = F.softmax(logits_all, dim=1).clamp(min=1e-6)
            weight_clinical = weight_all[:, :1]
            weight_patch_soft = weight_all[:, 1:]
            slide_feat = (
                weight_clinical * clinical_feat +
                torch.bmm(weight_patch_soft.unsqueeze(1), zp).squeeze(1)
            )
        else:
            weight_patch_soft = F.softmax(weight_logits_patch, dim=1).clamp(min=1e-6)
            slide_feat = torch.bmm(weight_patch_soft.unsqueeze(1), zp).squeeze(1)

        weight = self._minmax_norm(weight_patch_soft, dim=1)

        aux = {
            "attn_weight": attn_weight_patch,
            "u1": u1,
            "u2": u2,
            "weight": weight,
        }
        return slide_feat, aux

    def forward(
        self,
        inputs: torch.Tensor,
        clinical_feat: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.uncertainty_model(inputs, clinical_feat=clinical_feat)


class ClinicalTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-base-cased-v1.2",
        max_length: int = 256,
    ):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as e:
            raise ImportError(
                "transformers is required for clinical text encoding. "
                "Install it with: pip install transformers"
            ) from e

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = int(self.backbone.config.hidden_size)
        self.max_length = max_length

    def forward(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        toks = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        out = self.backbone(**toks)
        return out.last_hidden_state[:, 0, :]

class PatchEncoderTEAM(nn.Module):
    def __init__(
        self,
        enc_name: str,
        ckpt_path: str,
        backbone_model_name: str = "vit_large_patch16_224",
        img_size: int = 224,
        patch_size: int = 16,
        init_values: float = 1e-5,
        device: Optional[torch.device] = None,
        use_fp16: bool = False,
        feat_pool: str = "cls",
    ):
        super().__init__()
        self.enc_name = enc_name
        self.ckpt_path = ckpt_path
        self.backbone_model_name = backbone_model_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.init_values = init_values
        self.device = device
        self.use_fp16 = use_fp16
        self.feat_pool = feat_pool

        self.backbone = self._build_backbone()
        self._load_ckpt(self.ckpt_path)

    def _build_backbone(self) -> nn.Module:
        import timm
        uni_kwargs = dict(
            model_name=self.backbone_model_name,
            img_size=self.img_size,
            patch_size=self.patch_size,
            init_values=self.init_values,
            num_classes=0,
            dynamic_img_size=True
        )
        return timm.create_model(**uni_kwargs)

    @staticmethod
    def _strip_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("backbone."):
                k = k[len("backbone."):]
            if k.startswith("model."):
                k = k[len("model."):]
            new_sd[k] = v
        return new_sd

    def _load_ckpt(self, ckpt_path: str) -> None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[PatchEncoderTEAM] patch ckpt not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        if not isinstance(sd, dict):
            raise RuntimeError(f"[PatchEncoderTEAM] unexpected checkpoint format: {type(ckpt)}")

        sd = self._strip_prefix(sd)
        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)

        if len(unexpected) > 0:
            print(f"[PatchEncoderTEAM] WARN unexpected keys (first 20): {unexpected[:20]}")
        if len(missing) > 0:
            print(f"[PatchEncoderTEAM] WARN missing keys (first 20): {missing[:20]}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, N, C, H, W = x.shape
            x = x.reshape(B * N, C, H, W)
            feats = self._forward_4d(x)
            return feats.view(B, N, -1)
        if x.dim() == 4:
            return self._forward_4d(x)
        raise ValueError(f"[PatchEncoderTEAM] invalid input shape: {tuple(x.shape)}")

    def _forward_4d(self, x: torch.Tensor) -> torch.Tensor:
        if self.device is not None:
            x = x.to(self.device, non_blocking=True)
            self.backbone = self.backbone.to(self.device)

        self.backbone.eval()
        with torch.no_grad():
            if self.use_fp16 and x.is_cuda:
                with torch.cuda.amp.autocast():
                    feats = self._extract_feat(x)
            else:
                feats = self._extract_feat(x)
        return feats

    def _extract_feat(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone.forward_features(x)

        if isinstance(z, dict):
            z = z["x"] if "x" in z else next(v for v in z.values() if torch.is_tensor(v))

        if z.dim() == 3:
            return z.mean(dim=1) if self.feat_pool == "mean" else z[:, 0, :]
        return z

OutputMode = Literal["patch", "slide", "both"]

@dataclass
class EncoderConfig:
    patch_enc_name: str
    patch_ckpt: str
    slide_ckpt: str
    device: str = "cuda"
    use_fp16: bool = False
    patch_feat_dim: int = 1024
    slide_feat_dim: int = 512
    uncertainty_samples: int = 3
    dropout_rate: float = 0.3
    attn_hidden_dim: int = 256
    backbone_model_name: str = "vit_large_patch16_224"
    backbone_img_size: int = 224
    backbone_patch_size: int = 16
    backbone_init_values: float = 1e-5
    use_text: bool = False
    text_model_name: str = "dmis-lab/biobert-base-cased-v1.2"
    text_max_length: int = 256


class TEAMPathologyFeatureEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        dev = torch.device(cfg.device) if cfg.device else None

        self.patch_encoder = PatchEncoderTEAM(
            enc_name=cfg.patch_enc_name,
            ckpt_path=cfg.patch_ckpt,
            backbone_model_name=cfg.backbone_model_name,
            img_size=cfg.backbone_img_size,
            patch_size=cfg.backbone_patch_size,
            init_values=cfg.backbone_init_values,
            device=dev,
            use_fp16=cfg.use_fp16,
        )

        self.slide_encoder = SlideEncoderFeatOnly(
            patch_feat_dim=cfg.patch_feat_dim,
            slide_feat_dim=cfg.slide_feat_dim,
            uncertainty_samples=cfg.uncertainty_samples,
            dropout_rate=cfg.dropout_rate,
            attn_hidden_dim=cfg.attn_hidden_dim,
            finetune=False,
        )
        self._load_slide(cfg.slide_ckpt)
        self.use_text = cfg.use_text
        self.text_encoder = None
        self.text_proj = None
        if self.use_text:
            self.text_encoder = ClinicalTextEncoder(
                model_name=cfg.text_model_name,
                max_length=cfg.text_max_length,
            )
            self.text_proj = nn.Linear(self.text_encoder.hidden_size, cfg.slide_feat_dim)

        if dev is not None:
            self.slide_encoder = self.slide_encoder.to(dev)
            if self.text_encoder is not None:
                self.text_encoder = self.text_encoder.to(dev)
                self.text_proj = self.text_proj.to(dev)

    def _load_slide(self, ckpt_path: str) -> None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[TEAMPathologyFeatureEncoder] slide ckpt not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        if not isinstance(sd, dict):
            raise RuntimeError(f"[TEAMPathologyFeatureEncoder] unexpected slide ckpt format: {type(ckpt)}")

        keys = list(sd.keys())
        if len(keys) > 0 and all(k.startswith("module.") for k in keys):
            sd = {k[len("module."):]: v for k, v in sd.items()}

        missing, unexpected = self.slide_encoder.load_state_dict(sd, strict=False)
        if len(unexpected) > 0:
            print(f"[SlideEncoderFeatOnly] WARN unexpected keys (first 20): {unexpected[:20]}")
        if len(missing) > 0:
            print(f"[SlideEncoderFeatOnly] WARN missing keys (first 20): {missing[:20]}")

    def forward(
        self,
        patch_images: torch.Tensor,
        output_mode: OutputMode = "both",
        return_aux: bool = False,
        clinical_texts: Optional[Sequence[str]] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:

        out: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]] = {}

        patch_feat = self.patch_encoder(patch_images)
        if output_mode in ("patch", "both"):
            out["patch_feat"] = patch_feat

        if output_mode in ("slide", "both"):
            patch_feat_for_slide = patch_feat.unsqueeze(1) if patch_feat.dim() == 2 else patch_feat
            clinical_feat = None
            if self.use_text and clinical_texts is not None and len(clinical_texts) > 0:
                if len(clinical_texts) != patch_feat_for_slide.size(0):
                    raise ValueError(
                        f"clinical_texts length ({len(clinical_texts)}) must match batch size ({patch_feat_for_slide.size(0)})."
                    )
                text_vec = self.text_encoder(clinical_texts, patch_feat_for_slide.device)
                clinical_feat = self.text_proj(text_vec)
            slide_feat, aux = self.slide_encoder(patch_feat_for_slide, clinical_feat=clinical_feat)
            out["slide_feat"] = slide_feat
            if return_aux:
                out["aux"] = aux

        return out
