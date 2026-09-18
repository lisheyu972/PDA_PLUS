# PDA++: Field-Aligned Planning and Scene-Adaptive Insertion in Remote Sensing

[![arXiv](https://img.shields.io/badge/arXiv-2609.18329-b31b1b.svg)](https://arxiv.org/abs/2609.18329)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg)](https://arxiv.org/pdf/2609.18329v2)
[![ICML 2026](https://img.shields.io/badge/ICML-2026-4b44ce.svg)](https://openreview.net/forum?id=ojx0CyGXHJ)

This repository contains the **official implementation** of:

> **PDA++: Field-Aligned Planning and Scene-Adaptive Insertion in Remote Sensing**  
> Xianchi Dong, Yingyan Hou, Chao Ren, Wanxuan Lu, Zihan Wei, Hongfeng Yu, Yixiao Wang, Chubo Deng, and Xian Sun  
> **arXiv:2609.18329**, 2026

📄 **Paper:** [arXiv](https://arxiv.org/abs/2609.18329) | [PDF (v2)](https://arxiv.org/pdf/2609.18329v2)

PDA++ is an extended journal version of our ICML 2026 work:

> **Plan, Decouple, Assimilate: Physics-Aware Object Insertion in Remote Sensing Imagery**  
> Yingyan Hou, Xianchi Dong, Chao Ren, Wanxuan Lu, Zihan Wei, Hongfeng Yu, Yixiao Wang, and Xian Sun  
> **ICML 2026**

📄 **Conference Paper:** [OpenReview](https://openreview.net/forum?id=ojx0CyGXHJ)

---

## Overview

PDA++ is a unified framework for realistic object insertion in remote sensing imagery. It follows the **Plan–Decouple–Assimilate** paradigm and improves scene-adaptive insertion through field-aligned planning and environment-aware generation.

The framework contains three main stages:

- **Plan.** Determine scene-compatible object poses using an affordance field that jointly considers geometric clearance, structural cues, and target scale.
- **Decouple.** Construct a pose-conditioned background that provides explicit spatial guidance and target-scene context while preserving the identity of the reference object.
- **Assimilate.** Improve local appearance consistency by aligning multi-scale texture distributions between the inserted object and the surrounding scene.

PDA++ supports realistic remote sensing object insertion and can be used for downstream data augmentation in tasks such as object detection and semantic segmentation.

---

## Repository Structure

```text
PDA_PLUS/
├── Plan/          # Field-aligned planning and pose generation
└── generation/    # Scene-adaptive object insertion and generation
```

The code for planning and image generation is organized separately in the corresponding directories.

---


## Citation

If you find this repository useful for your research, please cite our PDA++ paper:

```bibtex
@misc{dong2026pdafieldalignedplanningsceneadaptive,
      title={PDA++: Field-Aligned Planning and Scene-Adaptive Insertion in Remote Sensing}, 
      author={Xianchi Dong and Yingyan Hou and Chao Ren and Wanxuan Lu and Zihan Wei and Hongfeng Yu and Yixiao Wang and Chubo Deng and Xian Sun},
      year={2026},
      eprint={2609.18329},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.18329},
}
```

Please also consider citing our ICML 2026 conference paper:

```bibtex
@inproceedings{hou2026plan,
      title={Plan, Decouple, Assimilate: Physics-Aware Object Insertion in Remote Sensing Imagery},
      author={Yingyan Hou and Xianchi Dong and Chao Ren and Wanxuan Lu and Zihan Wei and Hongfeng Yu and Yixiao Wang and Xian Sun},
      booktitle={Forty-third International Conference on Machine Learning},
      year={2026},
      url={https://openreview.net/forum?id=ojx0CyGXHJ},
}
```

---

## Acknowledgement

We thank the open-source community for the models, datasets, and tools that support research in remote sensing image generation and editing.


