# PDA_PLUS Generation

This directory contains the PDA_PLUS Generation module, including FLUX.2 LoRA training for object insertion, latent MATA inference, and the required `flux2` source code. Datasets, model weights, LoRA checkpoints, and generated outputs are not included in the repository.

## Directory Layout

```text
generation/
├── scripts/gen_train/
├── scripts/gen_infer/
├── src/flux2/
├── data/
├── models/
├── checkpoints/
└── output/
```

Place the base model weights at the following default locations:

```text
models/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors
models/FLUX.2-ae/ae.safetensors
```

## Inference

```bash
bash scripts/gen_infer/run_inference_insertion_seg_mata_latent.sh
```

## Training

```bash
PYTHONPATH=src python scripts/gen_train/train_lora_insertion_seg.py
```

All default paths are resolved relative to the `generation` directory. They can be overridden with command-line arguments or environment variables.

## Acknowledgements

This project is built upon the official FLUX.2 implementation developed by Black Forest Labs. We thank the authors for releasing their code and models.

Original repository:
https://github.com/black-forest-labs/flux2
