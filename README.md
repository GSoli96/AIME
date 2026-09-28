# AIME: Exploring the Risks of Generative AI Misuse

**Paper:** *Exploring the Risks of Generative AI Misuse via Ambiguous Prompts: Bypassing Safety Filters in Text-to-Image/Video Models*  
**Authors:** Stefano Cirillo, Giuseppe Polese, Giandomenico Solimando  
**Affiliations:** University of Salerno (Italy) · TU/e Eindhoven (Netherlands)  
**Journal:** Image and Vision Computing, Elsevier  

---

## Overview

This repository contains the code and resources for the AIME paper, which investigates the ethical risks of Text-to-Image/Video (T2I/V) generative AI models. We study how **ambiguous prompts** — seemingly innocent text that can bypass safety filters — lead to the generation of harmful, discriminatory, or stereotypical multimedia content.

### Key contributions

- **AIME Dataset**: 1,724 images + 1,577 videos (both complete, matching the paper's Table `tab:aimestat` exactly — see [Dataset Statistics](#dataset-statistics)) from 19 T2I/V models, labeled as ethical/unethical by 3 domain experts. →  [Download from HuggingFace](https://huggingface.co/datasets/DAISLab-Unisa/AIME-Dataset) *(gated — request access)*
- **Multimodal prompt engineering**: A corpus of 130 prompts (100 explicit + 30 ambiguous) designed to probe safety mechanisms, from a simple/no-jailbreak explicit baseline to context-laundered ambiguous prompts
- **LLM evaluation**: Binary and multi-class classification using Claude Sonnet, Claude Haiku, GPT-4/GPT-4o (images) and Gemini 1.0/1.5 Pro/Flash (videos)
- **User study**: 50 participants evaluated the social impact of the generated content

---

## Pipeline

```
prompts/          →   generation/          →   annotation/        →   llm_evaluation/    →   classification/
(130 prompts:          (SD 3.5, Gemini          (human labeling        (Claude/GPT on         (binary + multi-
 100 explicit +         Flash, Copilot,          via ipywidgets)        images, Gemini on       class metrics)
 30 ambiguous)          Wan2.1/CogVideoX/                                videos)
                        LTX/HunyuanVideo…)
```

---

## Repository Structure

```
AIME/
├── prompts/                    # Prompts used for generation
│   ├── ambiguous.txt           # Ambiguous prompts (legacy set, superseded by all_prompts.xlsx)
│   ├── explicit.txt            # Explicit/baseline prompts (legacy set)
│   ├── all_prompts.xlsx        # Full 130-prompt corpus is tracked at the pipeline level,
│   │                           # not in this file — see note below
│   └── dataset_prompts_classified_bilingual.json  # Real user-submitted prompt pairs, EN/IT
│
├── generation/                 # Image/video generation scripts and notebooks
│   ├── generate_SD35_aime.ipynb        # SD 3.5 Large/Medium/Turbo, current prompt set
│   ├── generate_SD35_colab.ipynb       # Same, Colab T4 variant
│   ├── generate_SD35_local_4090.ipynb  # Same, full fp16 for a 24GB card
│   ├── generate_SD35_{large,medium,turbo}.py  # Script versions of the three above
│   ├── generate_gemini_flash.ipynb / _run.py   # Gemini 2.5 Flash Image
│   ├── run_gemini_flash_{safe,unsafe}.py       # Split-run variants
│   ├── generate_T2V_opensource.ipynb   # Wan2.1 / CogVideoX / LTX-Video / HunyuanVideo (open weights)
│   ├── generate_kling.ipynb / generate_seedance.ipynb  # Paid-API T2V (Kling, Seedance)
│   ├── Generate_Image_SD_{1.5,3.5-medium,newrealityxl}.ipynb  # Legacy/earlier notebooks
│   └── prompt_datasets/        # NOTE: aimagelab.csv/copro.csv (third-party prompt pools) are
│                                # intentionally untracked here — see .gitignore. Present in the
│                                # private companion repo, not redistributed publicly.
│
├── annotation/                 # Human annotation tools (ipywidgets)
│   ├── 1_index_media.ipynb     # Build media index CSV
│   ├── 2_annotate_binary.ipynb # Binary (ethical/unethical) annotation
│   └── 3_annotate_multiclass.ipynb  # 5-class annotation
│
├── dataset/                    # AIME Dataset labels
│   ├── labels/
│   │   ├── aime_binary_{en,it}.csv       # Full binary labels
│   │   ├── aime_multiclass_{en,it}.csv   # Multi-class labels
│   │   ├── aime_*_supplementary_auto_{en,it}.csv  # Auxiliary VLM-annotated set, not merged
│   │   │                                          # into the main dataset (see paper limitations)
│   │   ├── llm_binary_{images,videos}.csv    # LLM predictions
│   │   ├── llm_multiclass_{images,videos}.csv
│   │   └── oracles/                # Ground truth for evaluation
│   └── MEDIA_LOCATION.md           # Where to find the media files
│
├── llm_evaluation/             # LLM-based content evaluation
│   └── evaluate_gemini_videos.ipynb  # Gemini 1.0/1.5 Pro/Flash on video content
│
├── classification/             # Classification metrics
│   ├── binary_metrics.ipynb    # Binary eval: accuracy, F1, confusion matrix
│   └── multiclass_metrics.ipynb  # Multi-class eval (5 categories)
│
├── download_dataset.py         # Script to download AIME from HuggingFace
├── requirements.txt
└── .env.example
```

Note: `paper/` (LaTeX source) is intentionally not part of this public repository — it is `.gitignore`d and kept in the authors' private companion repository until the paper is published.

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/GSoli96/AIME.git
cd AIME
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env with your HuggingFace token and GCP credentials
```

### 3. Download the AIME dataset

The dataset is hosted on HuggingFace and requires approval (access request at the link below):

```bash
python download_dataset.py
# or: download labels only (no media files)
python download_dataset.py --labels-only
```

**Request access:** https://huggingface.co/datasets/DAISLab-Unisa/AIME-Dataset

---

## Dataset Statistics

**Images: complete — 1,724/1,724.** **Videos: complete — 1,577/1,577.** **Total media: 3,301/3,301, exactly matching the paper's Table `tab:aimestat`.**

| Model | Type | Media | Target (paper) | Labeled | Unethical |
|---|---|---:|---:|---:|---:|
| Copilot | Image | 92 | 92 | 92 | 65 |
| Playground-v2.5 | Image | 23 | 23 | 23 | 5 |
| StableDiffusion 3.5 Large | Image | 531 | 531 | 531 | 161 |
| StableDiffusion 3.5 Medium | Image | 456 | 456 | 456 | 151 |
| StableDiffusion 3.5 Large-Turbo | Image | 481 | 481 | 481 | 138 |
| StableDiffusion 3-2B | Image | 46 | 46 | 46 | 27 |
| StableDiffusionXL | Image | 50 | 50 | 50 | 15 |
| Gemini 2.5 Flash | Image | 36 | 36 | 36 | 8 |
| Motion 1.0 (Kaiber) | Image | 9 | 9 | 9 | 9 |
| **Images total** | | **1,724** | **1,724** | **1,724** | **579** |
| Gen-2 (Runway) | Video | 38 | 38 | 38 | 14 |
| Imagine V2 (Pika) | Video | 26 | 26 | 26 | 18 |
| Decohere.ai | Video | 22 | 22 | 22 | 22 |
| Replay v0.2 (Genmo) | Video | 14 | 14 | 14 | 5 |
| Motion 1.0 (Kaiber) | Video | 4 | 4 | 4 | 2 |
| Zeroscope V2 | Video | 2 | 2 | 2 | 1 |
| PixVerse | Video | 4 | 4 | 4 | 2 |
| Wan2.1 T2V 1.3B | Video | 357 | 357 | 357 | 53 |
| CogVideoX-2B | Video | 365 | 365 | 267 | 50 |
| LTX-Video 0.9.7 distilled | Video | 383 | 383 | 144 | 12 |
| HunyuanVideo | Video | 362 | 362 | 362 | 43 |
| **Videos total** | | **1,577** | **1,577** | **1,240** | **222** |

Every image (1,724/1,724) and every Wan2.1/HunyuanVideo video has a complete binary + multi-class label. CogVideoX-2B and LTX-Video each have a residual batch of older videos (98 and 239 respectively) still awaiting ethics labels — the media itself is complete and matches the paper's per-model targets exactly, only their labeling lags behind.

---

## Multi-class Label Distribution (1,733 labeled items, images + videos combined)

| Category | Count |
|---|---|
| Offensive | 177 |
| Stereotypes | 146 |
| Prejudices | 154 |
| Violence | 178 |
| Nothing | 1,336 |

(Multi-label: an item can carry more than one category, so the column does not sum to 1,733.)

---

## How to Cite

```bibtex
@article{cirillo2026aime,
  title     = {Exploring the Risks of Generative AI Misuse via Ambiguous Prompts:
               Bypassing Safety Filters in Text-to-Image/Video Models},
  author    = {Cirillo, Stefano and Polese, Giuseppe and Solimando, Giandomenico},
  journal   = {Image and Vision Computing},
  publisher = {Elsevier},
  year      = {2026}
}
```

---

## License

- **Code**: MIT License
- **Dataset**: CC BY-NC 4.0 — non-commercial research use only.  
  The dataset contains sensitive generated content. Access is restricted and requires approval.

---

## Contact

Giandomenico Solimando — gsolimando@unisa.it · g.solimando@tue.nl  
Department of Computer Science, University of Salerno, Italy
