# TEAM

## 📄 Paper

**An Interactive Trustworthy AI Pathology Copilot to Improve Biomarker-Driven Prognostic Stratification and Therapeutic Response Prediction**

Yixiao Mao, Chengjie Xie, Feng Li, Danyi Li, Wenyan Zhang, Yidan Zhang, Bingbing Li, Chenglong Zhao, Zhengyu Zhang, Ying Tan, Zhijian Cen, Haisu Tao, Jian Yang, Jian Wang, Qianjin Feng, Boxiang Liu, Li Liang, Cheng Lu, Yu Zhang and Zhenyuan Ning.

The official implementation of TEAM. This project is based on the following great projects:

- CLAM
- DINOv2



## 🧭 Overview

This repository provides:

- Upstream TEAM feature extraction (patch-level and slide-level)
- Optional clinical text encoding in upstream aggregation (default off; enabled only when text is provided)
- Downstream Biomarker-driven TEAM model for outcome prediction

## 🗂️ Project Structure

```text
.
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- configs/
|   `-- team_config.json
|-- scripts/
|   |-- run_team.py
|   `-- run_biomarker_driven_team.py
|-- team/
|   |-- __init__.py
|   |-- config.py
|   |-- dataset.py
|   |-- utils.py
|   |-- patho_team_encoder.py
|   `-- biomarker_driven_team.py
|-- patch_weight.pth
`-- slide_weight.pth
```

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

## 🧱 Model Weights

Official TEAM checkpoints are released at:

- https://drive.google.com/drive/folders/1tDbM1GanVYa09wrDsyaqL-F8MmmIlJr5?usp=drive_link

Please download and place the files in the project root with the same names:

- `patch_weight.pth`
- `slide_weight.pth`

Default config already points to these names in `configs/team_config.json`:

- `upstream.paths.patch_ckpt = ./patch_weight.pth`
- `upstream.paths.slide_ckpt = ./slide_weight.pth`

If you store them in another folder, only update these two paths in `configs/team_config.json`.

## 🚀 How to Use TEAM

All key parameters are centralized in `configs/team_config.json`:

- upstream model dimensions and uncertainty settings
- upstream patch-backbone settings (model/img_size/patch_size/init_values)
- upstream runtime and path settings
- downstream model dimensions/classes and switches
- downstream runtime and checkpoint paths

You can change model dimensions, class counts, batch sizes, and defaults by editing this one file.

### Version A: without clinical text encoder (default)

In this mode, no clinical text vector is concatenated, and no clinical-text attention is used.

Single slide:

```bash
python scripts/run_team.py \
  --config ./configs/team_config.json \
  --input ./slide_patch_folder \
  --output ./slide_feat.pt \
  --output_mode both \
  --patch_ckpt ./patch_weight.pth \
  --slide_ckpt ./slide_weight.pth \
  --device cuda
```

Batch slides:

```bash
python scripts/run_team.py \
  --config ./configs/team_config.json \
  --input ./slides_root \
  --output_dir ./out_slide_feat \
  --batch_slides \
  --output_mode both \
  --patch_ckpt ./patch_weight.pth \
  --slide_ckpt ./slide_weight.pth \
  --device cuda
```

### Version B: with clinical text encoder (upstream only)

Single slide:

```bash
python scripts/run_team.py \
  --config ./configs/team_config.json \
  --input ./slide_patch_folder \
  --output ./slide_feat.pt \
  --output_mode both \
  --clinical_text "62-year-old male, stage III colorectal adenocarcinoma" \
  --use_text
```

Batch slides (`slide_name -> text` JSON):

```bash
python scripts/run_team.py \
  --config ./configs/team_config.json \
  --input ./slides_root \
  --output_dir ./out_slide_feat \
  --batch_slides \
  --output_mode both \
  --text_json ./slide_texts.json \
  --use_text
```


## 🔬 Downstream Tasks Evaluation

### 📊 Survival Analysis

The downstream module is named **Biomarker-driven TEAM**:

- file: `biomarker_driven_team.py`
- class: `BiomarkerDrivenTEAMModel`


```python
import torch
from team.biomarker_driven_team import BiomarkerDrivenTEAMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pack = torch.load("out_slide_feat/slide_A.pt", map_location="cpu")
feat_tensor = pack["patch_feat"].unsqueeze(0).to(device)  # [1, N, 1024]
cancer_id = torch.tensor([0], dtype=torch.long, device=device)

model = BiomarkerDrivenTEAMModel.from_config(
    "configs/team_config.json",
    device=device,
).to(device).eval()

with torch.no_grad():
    pred, fused_feat = model(feat_tensor, cancer_id)
```

Run script:

```bash
python scripts/run_biomarker_driven_team.py \
  --config ./configs/team_config.json \
  --input_pt ./out_slide_feat/slide_A.pt \
  --device cuda \
  --use_stage --use_tme
```

## 📚 Citation

```bibtex
@article{team_pathology_2026,
  title   = {An Interactive Trustworthy AI Pathology Copilot to Improve Biomarker-Driven Prognostic Stratification and Therapeutic Response Prediction},
  author  = {Mao, Yixiao and Xie, Chengjie and Li, Feng and others},
  journal = {Under Review},
  year    = {2026}
}
```
