# Assignment 1 — Blood Cell Image Classification

## Task Overview

This assignment focuses on solving a **multi-class image classification problem** using deep neural networks. The goal is to correctly classify 96x96 RGB images of blood cells into 8 distinct cell types.

---

## Dataset Description

- **Image size**: 96 x 96
- **Color channels**: RGB (3 channels)
- **Total samples**: 13,759 images
- **File format**: `.npz` (NumPy archive)
- **Input shape**: (96, 96, 3)
- **Classes (0–7)**:
  | Label | Cell Type               |
  |-------|--------------------------|
  | 0     | Basophil                 |
  | 1     | Eosinophil               |
  | 2     | Erythroblast             |
  | 3     | Immature granulocytes    |
  | 4     | Lymphocyte               |
  | 5     | Monocyte                 |
  | 6     | Neutrophil               |
  | 7     | Platelet                 |

You can load the dataset using:
```python
import numpy as np
data = np.load('training_set.npz')
X = data['images']
y = data['labels']
