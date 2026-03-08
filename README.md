# LFM2 — Liquid Foundation Models

PyTorch implementation of the **LFM2 Technical Report** ([arXiv:2511.23404](https://arxiv.org/abs/2511.23404)) by Liquid AI.

LFM2 is a family of Liquid Foundation Models designed for efficient on-device deployment with strong task capabilities. The architecture uses a minimal hybrid backbone combining gated short convolutions with grouped query attention (GQA) blocks.

## Quick Start

```bash
pip install -e .
```

## Usage

### 1. Initialization
```python
from lfm2.model.configs import LFM2_350M
from lfm2.model.backbone import LFM2Model

config = LFM2_350M
model = LFM2Model(config)
```

### 2. Inference
Use the provided generation script:
```bash
python scripts/generate.py --model lfm2-350m --prompt "Hello LFM2"
```

### 3. Training with Distillation
```python
from lfm2.training.trainer import LFM2Trainer
trainer = LFM2Trainer(model, teacher_model=teacher)
metrics = trainer.train_step(input_ids, labels)
```

### 4. Multimodal (LFM2-VL)
```python
from lfm2.model.vlm import LFM2VL
vlm = LFM2VL(model)
# pixel_values should be preprocessed by ImagePreprocessor
output = vlm(input_ids, pixel_values=pixel_values)
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
