# LFM2 — Liquid Foundation Models

PyTorch implementation of the **LFM2 Technical Report** ([arXiv:2511.23404](https://arxiv.org/abs/2511.23404)) by Liquid AI.

LFM2 is a family of Liquid Foundation Models designed for efficient on-device deployment with strong task capabilities. The architecture uses a minimal hybrid backbone combining gated short convolutions with grouped query attention (GQA) blocks.

## Quick Start

```bash
pip install -e .
```

## Project Structure

```
lfm2/
  model/       — Architecture components (backbone, attention, convolution, MoE)
  training/    — Training infrastructure (distillation, alignment, merging)
  utils/       — Utility functions
tests/         — Unit and integration tests
scripts/       — Training and inference entry points
examples/      — Usage examples
```

## Citation

```bibtex
@article{lfm2,
  title={LFM2 Technical Report},
  author={Amini, Alexander and Banaszak, Anna and Benoit, Harold and others},
  journal={arXiv preprint arXiv:2511.23404},
  year={2025}
}
```
