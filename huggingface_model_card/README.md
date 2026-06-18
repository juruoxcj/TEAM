---
license: apache-2.0
library_name: pytorch
pipeline_tag: image-feature-extraction
tags:
  - computational-pathology
  - histopathology
  - whole-slide-imaging
  - breast-cancer
  - biomarker-driven-ai
  - feature-extraction
  - pytorch
  - team
model-index:
  - name: TEAM
    results: []
---

# TEAM

TEAM is a PyTorch model release for computational pathology feature extraction and biomarker-driven downstream prediction. It contains a patch-level pathology encoder and a slide-level TEAM aggregation checkpoint for whole-slide-image-derived patch folders.

The accompanying code is released at: https://github.com/juruoxcj/TEAM

## Model Files

| File | Description |
| --- | --- |
| `patch_weight.pth` | Patch-level pathology encoder checkpoint. |
| `slide_weight.pth` | Slide-level TEAM aggregation checkpoint. |

Place both files in the root of the GitHub repository checkout, or pass their paths explicitly through the inference command.

## Intended Use

TEAM is intended for research use in computational pathology workflows, including:

- extracting patch-level pathology embeddings from image tiles;
- aggregating patch features into slide-level TEAM representations;
- supporting downstream biomarker-driven prognostic stratification and therapeutic response prediction experiments when paired with the released source code.

TEAM is not intended for standalone clinical diagnosis, treatment selection or deployment as a medical device.

## Quick Start

Install the code from GitHub:

```bash
git clone https://github.com/juruoxcj/TEAM.git
cd TEAM
conda env create -f environment.yml
conda activate team
```

Download `patch_weight.pth` and `slide_weight.pth` from this model repository and run the included demo:

```bash
python run_team.py \
  --config ./configs/demo_config.json \
  --patch_ckpt ./patch_weight.pth \
  --slide_ckpt ./slide_weight.pth \
  --device cuda
```

The demo uses six unmodified TCGA BRCA patch images included in the GitHub repository. On the tested CUDA workstation, the demo completed in about 21 seconds.

## Inputs

TEAM expects one folder per slide, containing pre-extracted image patches:

```text
slides/
`-- slide_A/
    |-- patch_0001.png
    |-- patch_0002.png
    `-- ...
```

Supported image extensions are `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp` and `.webp`.

Optional de-identified clinical text can be provided as a JSON file mapping slide folder names to text. Do not include protected health information in filenames, metadata or text fields.

## Outputs

The upstream inference script writes a PyTorch `.pt` file containing:

| Key | Shape | Description |
| --- | --- | --- |
| `patch_feat` | `[N, 1024]` | Patch-level embeddings for `N` input patches. |
| `slide_feat` | `[1, 512]` | Slide-level TEAM embedding. |
| `paths` | length `N` | Input patch paths in feature order. |

## Tested Environment

The public demo was verified with:

- Python 3.10.18
- PyTorch 2.7.1+cu126
- CUDA runtime 12.6
- NVIDIA RTX 5880 Ada Generation GPU
- Linux 5.15 x86_64

CPU execution is suitable for installation checks and downstream smoke tests. GPU execution is recommended for upstream feature extraction.

## Training Data and Evaluation

TEAM was developed for pathology image analysis using whole-slide-image-derived patch data and associated study metadata described in the accompanying manuscript. Raw WSIs, patient-derived patch tiles, clinical records, patient-level annotations and generated feature tensors are not redistributed in this model repository.

Quantitative evaluation should be interpreted through the accompanying manuscript and its Data Availability statement. The small public demo patches are provided only to demonstrate the input layout and inference workflow; they are not an evaluation dataset.

## Limitations and Ethics

- The released checkpoints are for research use.
- Performance can vary across scanners, staining protocols, tissue preparation pipelines and patient cohorts.
- Clinical deployment requires independent validation, governance review and compliance with local regulations.
- Users are responsible for following data-use agreements and privacy requirements for controlled pathology data.

## License

The model card and associated code release use the Apache License 2.0. Checkpoint use should follow this license and any applicable data-use restrictions described by the authors.

## Citation

```bibtex
@article{team_pathology_2026,
  title   = {An Interactive Trustworthy AI Pathology Copilot to Improve Biomarker-Driven Prognostic Stratification and Therapeutic Response Prediction},
  author  = {Mao, Yixiao and Xie, Chengjie and Li, Feng and others},
  journal = {Under Review},
  year    = {2026}
}
```
