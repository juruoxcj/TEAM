# -*- coding: utf-8 -*-
"""
Biomarker-driven TEAM for downstream outcome prediction.

Expected upstream input: feat_tensor [B, N, D_patch].
"""

from __future__ import annotations
from typing import List, Optional
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from team.patho_team_encoder import SlideEncoderFeatOnly
from team.config import load_team_config


def _load_state(model: nn.Module, ckpt_path: Optional[str], device: torch.device, tag: str) -> None:
    if not ckpt_path or not os.path.exists(ckpt_path):
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(sd, dict):
        raise RuntimeError(f"[{tag}] unexpected checkpoint format: {type(ckpt)}")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[{tag}] WARN missing keys (first 20): {missing[:20]}")
    if unexpected:
        print(f"[{tag}] WARN unexpected keys (first 20): {unexpected[:20]}")
    print(f"[Info] Loaded {tag} weights from {ckpt_path}")


class SlideTaskModel(nn.Module):
    def __init__(
        self,
        n_classes: int,
        patch_feat_dim: int = 1024,
        slide_feat_dim: int = 512,
        uncertainty_samples: int = 3,
        dropout_rate: float = 0.3,
        attn_hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = SlideEncoderFeatOnly(
            patch_feat_dim=patch_feat_dim,
            slide_feat_dim=slide_feat_dim,
            uncertainty_samples=uncertainty_samples,
            dropout_rate=dropout_rate,
            attn_hidden_dim=attn_hidden_dim,
            finetune=False,
        )
        self.fc = nn.Linear(slide_feat_dim, n_classes)

    def uncertainty_model(self, feat_tensor: torch.Tensor):
        slide_feat, aux = self.encoder.uncertainty_model(feat_tensor)
        u1 = aux["u1"]
        u2 = aux["u2"]
        branch_uncertainty = torch.mean(u1 + u2, dim=1)
        weight = aux["weight"] if "weight" in aux else torch.ones_like(u1)
        return (
            slide_feat,
            aux["attn_weight"],
            u1,
            u2,
            branch_uncertainty,
            weight,
        )

    def forward(self, feat_tensor: torch.Tensor) -> torch.Tensor:
        return self.fc(self.uncertainty_model(feat_tensor)[0])


class BiomarkerDrivenTEAMModel(nn.Module):
    """Biomarker-driven Outcome Prediction model."""

    def __init__(
        self,
        device: torch.device,
        slide_ckpt: Optional[str] = None,
        use_stage: bool = True,
        stage_ckpt: Optional[str] = None,
        use_tme: bool = False,
        tme_ckpt: Optional[str] = None,
        use_gene: bool = False,
        gene_ckpt: Optional[str] = None,
        patch_feat_dim: int = 1024,
        shared_dim: int = 512,
        stage_classes: int = 4,
        tme_classes: int = 8,
        gene_classes: int = 15717,
        uncertainty_samples: int = 3,
        dropout_rate: float = 0.3,
        attn_hidden_dim: int = 256,
        num_cancers: int = 32,
        tau: float = 1.1,
    ):
        super().__init__()
        self.device = device
        self.use_stage = use_stage
        self.use_tme = use_tme
        self.use_gene = use_gene
        self.patch_feat_dim = patch_feat_dim
        self.shared_dim = shared_dim
        self.stage_classes = stage_classes
        self.tme_classes = tme_classes
        self.gene_classes = gene_classes
        self.tau = tau

        self.cancer_emb = nn.Embedding(num_embeddings=num_cancers, embedding_dim=self.shared_dim)

        self.slide_model = SlideTaskModel(
            n_classes=1,
            patch_feat_dim=self.patch_feat_dim,
            slide_feat_dim=self.shared_dim,
            uncertainty_samples=uncertainty_samples,
            dropout_rate=dropout_rate,
            attn_hidden_dim=attn_hidden_dim,
        ).to(device)
        _load_state(self.slide_model, slide_ckpt, device, "slide")

        if self.use_stage:
            self.stage_model = SlideTaskModel(
                n_classes=self.stage_classes,
                patch_feat_dim=self.patch_feat_dim,
                slide_feat_dim=self.shared_dim,
                uncertainty_samples=uncertainty_samples,
                dropout_rate=dropout_rate,
                attn_hidden_dim=attn_hidden_dim,
            ).to(device)
            _load_state(self.stage_model, stage_ckpt, device, "stage")
            for p in self.stage_model.parameters():
                p.requires_grad = False
            self.stage_proj = nn.Linear(self.stage_classes, self.shared_dim)
            self.attn_stage = nn.Linear(self.shared_dim, 1)

        if self.use_tme:
            self.tme_model = SlideTaskModel(
                n_classes=self.tme_classes,
                patch_feat_dim=self.patch_feat_dim,
                slide_feat_dim=self.shared_dim,
                uncertainty_samples=uncertainty_samples,
                dropout_rate=dropout_rate,
                attn_hidden_dim=attn_hidden_dim,
            ).to(device)
            _load_state(self.tme_model, tme_ckpt, device, "tme")
            for p in self.tme_model.parameters():
                p.requires_grad = False
            self.tme_proj = nn.Linear(self.tme_classes, self.shared_dim)
            self.attn_tme = nn.Linear(self.shared_dim, 1)

        if self.use_gene:
            self.gene_model = SlideTaskModel(
                n_classes=self.gene_classes,
                patch_feat_dim=self.patch_feat_dim,
                slide_feat_dim=self.shared_dim,
                uncertainty_samples=uncertainty_samples,
                dropout_rate=dropout_rate,
                attn_hidden_dim=attn_hidden_dim,
            ).to(device)
            _load_state(self.gene_model, gene_ckpt, device, "gene")
            for p in self.gene_model.parameters():
                p.requires_grad = False
            self.gene_proj = nn.Linear(self.gene_classes, self.shared_dim)
            self.attn_gene = nn.Linear(self.shared_dim, 1)

        self.attn_slide = nn.Linear(self.shared_dim, 1)
        self.final_fc = nn.Linear(self.shared_dim, 1)

    def tme_dim_weights(self, feat_tensor: torch.Tensor, cancer_id: torch.Tensor):
        _ = cancer_id
        assert self.use_tme, "use_tme=False: TME branch is disabled."

        tme_logits = self.tme_model.fc(self.tme_model.uncertainty_model(feat_tensor)[0])
        A = self.attn_tme.weight
        P = self.tme_proj.weight
        k = (A @ P).squeeze(0)
        contrib = tme_logits * k.unsqueeze(0)
        weights = F.softmax(contrib, dim=1)
        return weights, contrib, tme_logits

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.shared_dim,))

    @classmethod
    def from_config(
        cls,
        config_path: str = "configs/team_config.json",
        device: Optional[torch.device] = None,
        overrides: Optional[dict] = None,
    ):
        cfg = load_team_config(config_path)
        d_model = cfg["downstream"]["model"]
        d_paths = cfg["downstream"]["paths"]
        d_runtime = cfg["downstream"]["runtime"]
        overrides = overrides or {}
        if device is None:
            dev_name = d_runtime.get("device", "cuda")
            if isinstance(dev_name, str) and dev_name.startswith("cuda") and not torch.cuda.is_available():
                dev_name = "cpu"
            device = torch.device(dev_name)
        params = dict(
            device=device,
            slide_ckpt=d_paths.get("slide_ckpt", None),
            use_stage=bool(d_model.get("use_stage", True)),
            stage_ckpt=d_paths.get("stage_ckpt", None),
            use_tme=bool(d_model.get("use_tme", False)),
            tme_ckpt=d_paths.get("tme_ckpt", None),
            use_gene=bool(d_model.get("use_gene", False)),
            gene_ckpt=d_paths.get("gene_ckpt", None),
            patch_feat_dim=int(d_model.get("patch_feat_dim", 1024)),
            shared_dim=int(d_model.get("shared_dim", 512)),
            stage_classes=int(d_model.get("stage_classes", 4)),
            tme_classes=int(d_model.get("tme_classes", 8)),
            gene_classes=int(d_model.get("gene_classes", 15717)),
            uncertainty_samples=int(cfg["upstream"]["model"].get("uncertainty_samples", 3)),
            dropout_rate=float(cfg["upstream"]["model"].get("dropout_rate", 0.3)),
            attn_hidden_dim=int(cfg["upstream"]["model"].get("attn_hidden_dim", 256)),
            num_cancers=int(d_model.get("num_cancers", 32)),
            tau=float(d_model.get("tau", 1.1)),
        )
        params.update(overrides)
        return cls(**params)

    def forward(
        self,
        feat_tensor: torch.Tensor,
        cancer_id: torch.Tensor,
    ):
        c_emb = self.cancer_emb(cancer_id)

        features: List[torch.Tensor] = []
        attn_scores: List[torch.Tensor] = []

        slide_out = self.slide_model.uncertainty_model(feat_tensor)
        slide_feat = slide_out[0]
        slide_feat = self._norm(slide_feat + c_emb)
        features.append(slide_feat)
        attn_scores.append(self.attn_slide(slide_feat))

        if self.use_stage:
            stage_out = self.stage_model.uncertainty_model(feat_tensor)
            stage_logits = self.stage_model.fc(stage_out[0])
            stage_probs = F.softmax(stage_logits, dim=1)
            stage_feat = self._norm(self.stage_proj(stage_probs) + c_emb)
            features.append(stage_feat)
            attn_scores.append(self.attn_stage(stage_feat))

        if self.use_tme:
            tme_out = self.tme_model.uncertainty_model(feat_tensor)
            tme_logits = self.tme_model.fc(tme_out[0])
            tme_feat = self._norm(self.tme_proj(tme_logits) + c_emb)
            features.append(tme_feat)
            attn_scores.append(self.attn_tme(tme_feat))

        if self.use_gene:
            gene_out = self.gene_model.uncertainty_model(feat_tensor)
            gene_output = self.gene_model.fc(gene_out[0])
            gene_feat = self._norm(self.gene_proj(gene_output) + c_emb)
            features.append(gene_feat)
            attn_scores.append(self.attn_gene(gene_feat))

        attn_scores = torch.cat(attn_scores, dim=1)
        scores = (attn_scores - attn_scores.mean(dim=1, keepdim=True)) / (
            attn_scores.std(dim=1, keepdim=True) + 1e-6
        )
        attn_weights = F.softmax(scores / self.tau, dim=1)

        fused_feat = torch.stack(features, dim=1)
        fused_feat = torch.sum(attn_weights.unsqueeze(2) * fused_feat, dim=1)

        pred = self.final_fc(fused_feat)
        return pred, fused_feat

MultiModalFusionModel = BiomarkerDrivenTEAMModel
