# TEAM

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.1-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![Model Card](https://img.shields.io/badge/HuggingFace-model%20card-yellow.svg)](huggingface_model_card/README.md)

🚀 Official implementation for:

**An Interactive Trustworthy AI Pathology Copilot to Improve Biomarker-Driven Prognostic Stratification and Therapeutic Response Prediction**

Yixiao Mao, Chengjie Xie, Feng Li, Danyi Li, Wenyan Zhang, Yidan Zhang, Bingbing Li, Chenglong Zhao, Zhengyu Zhang, Ying Tan, Zhijian Cen, Haisu Tao, Jian Yang, Jian Wang, Qianjin Feng, Boxiang Liu, Li Liang, Cheng Lu, Yu Zhang and Zhenyuan Ning.

TEAM provides a compact release for pathology feature extraction and downstream biomarker-driven prediction. The repository includes the source code, tested environment, demo patches, inference commands, expected outputs, model weight instructions, license and citation information.

**Code:** https://github.com/juruoxcj/TEAM

## 🔥 Highlights

- 🧬 Patch-level pathology encoder and slide-level TEAM aggregator.
- 🧠 Optional de-identified clinical-text conditioning for upstream aggregation.
- 📊 Downstream Biomarker-driven TEAM model for outcome prediction.
- 🧪 Six unmodified TCGA BRCA demo patches for a fast runnable example.
- 📦 Hugging Face model card template for `patch_weight.pth` and `slide_weight.pth`.

## 📌 News

- **2026-06:** Initial open-source release prepared with demo data, Apache-2.0 license, tested Python 3.10 environment and Hugging Face model card.

## 🖼️ Demo Patches

The demo folder contains six original TCGA BRCA patch images copied without color modification or re-encoding:

```text
demo_data/slides/TCGA-A7-A13G-01Z-00-DX1.C258C545-8C1F-41D4-846F-962A746CBDFB/
|-- patch_660.jpg
|-- patch_661.jpg
|-- patch_662.jpg
|-- patch_663.jpg
|-- patch_664.jpg
`-- patch_665.jpg
```

<p>
  <img src="demo_data/slides/TCGA-A7-A13G-01Z-00-DX1.C258C545-8C1F-41D4-846F-962A746CBDFB/patch_660.jpg" width="128" />
  <img src="demo_data/slides/TCGA-A7-A13G-01Z-00-DX1.C258C545-8C1F-41D4-846F-962A746CBDFB/patch_661.jpg" width="128" />
  <img src="demo_data/slides/TCGA-A7-A13G-01Z-00-DX1.C258C545-8C1F-41D4-846F-962A746CBDFB/patch_662.jpg" width="128" />
</p>

These patches are included only to demonstrate the expected input layout and run the software demo. They should not be used for scientific evaluation.

## 🗂️ Repository Layout

```text
.
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- environment.yml
|-- run_team.py
|-- configs/
|   |-- team_config.json
|   `-- demo_config.json
|-- demo_data/
|   |-- README.md
|   |-- slide_texts.demo.json
|   `-- slides/
|       `-- TCGA-A7-A13G-01Z-00-DX1.C258C545-8C1F-41D4-846F-962A746CBDFB/
|-- huggingface_model_card/
|   `-- README.md
|-- scripts/
|   |-- run_team.py
|   |-- run_biomarker_driven_team.py
|   `-- smoke_test.py
`-- team/
    |-- config.py
    |-- dataset.py
    |-- utils.py
    |-- patho_team_encoder.py
    `-- biomarker_driven_team.py
```

## ⚙️ System Requirements

TEAM is written in Python and can run on Linux, macOS and Windows when the required packages are installed. GPU feature extraction is expected to be run on Linux with CUDA.

| Item | Tested setting |
| --- | --- |
| OS | Linux 5.15.0-67-generic x86_64, glibc 2.31 |
| Python | 3.10.18 |
| PyTorch | 2.7.1+cu126 |
| CUDA runtime | 12.6 |
| GPU | NVIDIA RTX 5880 Ada Generation, 49 GB memory |

CPU is sufficient for installation checks and the no-data downstream smoke test. A CUDA-compatible NVIDIA GPU is recommended for upstream feature extraction. For full-size slide batches, reduce `upstream.runtime.batch_size` in `configs/team_config.json` if memory is limited.

## 🛠️ Installation

### Conda

```bash
conda env create -f environment.yml
conda activate team
```

### venv + pip

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU inference, install a PyTorch build matching your CUDA runtime before installing the remaining dependencies.

Typical installation time is **5-15 minutes** when compatible PyTorch/CUDA packages are already cached, and **15-45 minutes** when packages must be downloaded from scratch.

## 📦 Model Weights

TEAM uses two checkpoint files:

| File | Description |
| --- | --- |
| `patch_weight.pth` | patch-level pathology encoder checkpoint |
| `slide_weight.pth` | slide-level TEAM aggregation checkpoint |

The checkpoints are prepared for release on Hugging Face. The model card template is provided at [huggingface_model_card/README.md](huggingface_model_card/README.md). After creating the Hugging Face model repository, upload:

```text
patch_weight.pth
slide_weight.pth
README.md  # copy from huggingface_model_card/README.md
```

The default config expects root-level checkpoint names:

```text
upstream.paths.patch_ckpt = ./patch_weight.pth
upstream.paths.slide_ckpt = ./slide_weight.pth
```

If checkpoints are stored elsewhere, update `configs/team_config.json` or pass `--patch_ckpt` and `--slide_ckpt` on the command line.

## 🚀 Quick Start

### 1. No-data downstream smoke test

This check validates the downstream TEAM model graph with synthetic feature tensors. It does not require model weights, demo images or a GPU.

```bash
python scripts/smoke_test.py --device cpu --num_patches 8
```

Expected output:

```text
pred shape: (1, 1)
fused_feat shape: (1, 512)
[OK] TEAM downstream smoke test passed.
```

Expected run time on a normal desktop CPU is **less than 1 minute** after installation.

### 2. Upstream demo feature extraction

After downloading `patch_weight.pth` and `slide_weight.pth`, run:

```bash
python run_team.py \
  --config ./configs/demo_config.json \
  --patch_ckpt ./patch_weight.pth \
  --slide_ckpt ./slide_weight.pth \
  --device cuda
```

This writes:

```text
outputs/TCGA-A7-A13G-01Z-00-DX1.C258C545-8C1F-41D4-846F-962A746CBDFB.pt
```

Expected tensor contents:

| Key | Shape | Description |
| --- | --- | --- |
| `patch_feat` | `[6, 1024]` | patch-level embeddings |
| `slide_feat` | `[1, 512]` | slide-level TEAM embedding |
| `paths` | length 6 | input patch paths |

On the tested CUDA workstation, the six-patch demo completed in **about 21 seconds**. First-time model initialization and CPU-only execution can be slower.

## 🧪 Run TEAM on Your Data

Prepare one folder per slide:

```text
my_slides/
|-- slide_A/
|   |-- patch_0001.png
|   |-- patch_0002.png
|   `-- ...
`-- slide_B/
    |-- patch_0001.png
    |-- patch_0002.png
    `-- ...
```

Supported image extensions are `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp` and `.webp`.

Single-slide inference:

```bash
python run_team.py \
  --config ./configs/team_config.json \
  --input ./my_slides/slide_A \
  --output ./outputs/slide_A.pt \
  --output_mode both \
  --patch_ckpt ./patch_weight.pth \
  --slide_ckpt ./slide_weight.pth \
  --device cuda
```

Batch inference:

```bash
python run_team.py \
  --config ./configs/team_config.json \
  --input ./my_slides \
  --output_dir ./outputs/team_features \
  --batch_slides \
  --output_mode both \
  --patch_ckpt ./patch_weight.pth \
  --slide_ckpt ./slide_weight.pth \
  --device cuda
```

Optional de-identified clinical text can be supplied with a JSON file mapping slide folder names to text:

```json
{
  "slide_A": "De-identified clinical text for slide A",
  "slide_B": "De-identified clinical text for slide B"
}
```

```bash
python run_team.py \
  --config ./configs/team_config.json \
  --input ./my_slides \
  --output_dir ./outputs/team_features_with_text \
  --batch_slides \
  --output_mode both \
  --text_json ./slide_texts.json \
  --use_text \
  --patch_ckpt ./patch_weight.pth \
  --slide_ckpt ./slide_weight.pth \
  --device cuda
```

Do not include protected health information in filenames, metadata or clinical text.

## 📊 Downstream Prediction

The downstream module is `BiomarkerDrivenTEAMModel`.

```python
import torch
from team.biomarker_driven_team import BiomarkerDrivenTEAMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pack = torch.load("outputs/team_features/slide_A.pt", map_location="cpu")
feat_tensor = pack["patch_feat"].unsqueeze(0).to(device)
cancer_id = torch.tensor([0], dtype=torch.long, device=device)

model = BiomarkerDrivenTEAMModel.from_config(
    "configs/team_config.json",
    device=device,
).to(device).eval()

with torch.no_grad():
    pred, fused_feat = model(feat_tensor, cancer_id)
```

Command-line example:

```bash
python scripts/run_biomarker_driven_team.py \
  --config ./configs/team_config.json \
  --input_pt ./outputs/team_features/slide_A.pt \
  --device cuda \
  --use_stage
```

## 🔐 Data and Privacy

Raw WSIs, patient-derived patch tiles, clinical records, patient-level annotations and generated feature tensors may be controlled data. Except for the six selected demo patches under `demo_data/`, this GitHub repository only includes source code and configuration files. Use approved institutional and data-use procedures for controlled data access.

## 📄 License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE).

## ✏️ Citation

```bibtex
@article{team_pathology_2026,
  title   = {An Interactive Trustworthy AI Pathology Copilot to Improve Biomarker-Driven Prognostic Stratification and Therapeutic Response Prediction},
  author  = {Mao, Yixiao and Xie, Chengjie and Li, Feng and others},
  journal = {Under Review},
  year    = {2026}
}
```
