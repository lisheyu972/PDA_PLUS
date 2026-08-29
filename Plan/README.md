# PDA_PLUS Plan

This directory contains the PDA_PLUS Plan module, the SegEarthOV3 inference adapter, and the original third-party SAM3 source code. Datasets, model weights, and generated outputs are not included in the repository.

## Directory Layout

```text
Plan/
├── batch_Seg_AAP.py
├── segearthov3_segmentor.py
├── configs/mar20_names.txt
├── sam3/
├── data/MAR20_seg/train_sub/images/
├── data/MAR20_seg/train_sub/labelTxt/
├── models/sam3.pt
└── outputs/generated_labels/
```

The contents of `data/`, `models/`, and `outputs/` are ignored by Git. All runtime paths are resolved relative to the `Plan` directory and do not depend on the current working directory.

## Installation and Usage

```bash
python -m pip install -r requirements.txt
python batch_Seg_AAP.py
```

Before running the module, place the SAM3 checkpoint at `models/sam3.pt` and organize the input images and DOTA labels according to the directory layout above. Class names are configured in `configs/mar20_names.txt`, and generated results are written to `outputs/generated_labels/`.
