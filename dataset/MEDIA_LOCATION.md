# Media File Location

The media files (images and videos) are **not stored in this repository** due to their size.

## Download from HuggingFace

The full dataset (1,445 images + 110 videos) is hosted at:

**https://huggingface.co/datasets/DAISLab-Unisa/AIME-Dataset**

> Access is gated — request approval by filling out the form on the HuggingFace page.

To download:

```bash
python ../download_dataset.py
```

This will create a `dataset/media/` folder with the following structure:

```
dataset/media/
├── images/
│   ├── Copilot/          (92 files)
│   ├── Playground_v2.5/  (23 files)
│   ├── StableDiffusion3.5_Large/        (450 files)
│   ├── StableDiffusion3.5_Medium/       (375 files)
│   ├── StableDiffusion3.5_Large_Turbo/  (400 files)
│   ├── StableDiffusion3-2B/             (46 files)
│   ├── StableDiffusionXL/               (50 files)
│   └── Kaiber/           (9 files)
└── videos/
    ├── Runway/            (38 files)
    ├── Pika/              (26 files)
    ├── Decohere/          (22 files)
    ├── Genmo/             (14 files)
    ├── Kaiber/            (4 files)
    ├── Zeroscope/         (2 files)
    └── Pixverse/          (4 files)
```

## Label Files

The label CSV files are already included in `dataset/labels/` and reference files
by their relative path within `dataset/media/` (e.g., `images/Copilot/Copilot_0.jpeg`).
