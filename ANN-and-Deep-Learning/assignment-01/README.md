# Assignment 1 — Blood Cell Image Classification

## Task Overview

This assignment focuses on solving a **multi-class image classification problem** using deep neural networks. The goal is to correctly classify **96x96 RGB images of blood cells** into 8 distinct cell types.

---

## Dataset Description

- **Image size**: 96 x 96
- **Color channels**: RGB (3 channels)
- **Total images**: 11,959 images used for training, validation, and testing
- **Input shape**: (96, 96, 3)
- **Labels**: One-hot encoded into 8 classes

| Label | Cell Type |
|-------|----------------------------|
| 0     | Basophil                   |
| 1     | Eosinophil                 |
| 2     | Erythroblast               |
| 3     | Immature granulocytes      |
| 4     | Lymphocyte                 |
| 5     | Monocyte                   |
| 6     | Neutrophil                 |
| 7     | Platelet                   |

- **Challenges**:
  - High intra-class variation
  - Relatively small dataset
  - Potential class imbalance (confirmed and handled)

---

## Approach

### Data Preprocessing

- Invalid data (duplicates and "rick-rolled" images) filtered (~15% of dataset).
- Uniform scaling normalization: pixel values converted from `[0, 255]` to `[0, 1]`.

### Image Augmentation

- Rotation (full 360°)
- Width and height shift
- Shearing
- Zooming
- Horizontal flipping
- Brightness change was tested but discarded (negative effect on critical cell features).

### Class Imbalance Handling

- Class weights applied to the loss function to improve performance on underrepresented classes (0, 4, 5).
- Resulted in significant validation improvement for minority classes.

---

## Experiments

### 1️⃣ Traditional CNN

- 3 convolutional blocks + fully connected layer
- Batch normalization and dropout
- Softmax output
- Local accuracy ~90%, online test accuracy only 15% → lacked generalization.

### 2️⃣ DenseNet169 Model

- Pretrained DenseNet169 with ImageNet weights
- GlobalAveragePooling2D + batch normalization + dropout + dense layer (L1/L2 regularization)
- Online test accuracy improved to 47%.

### 3️⃣ ConvNeXtXLarge Pretrained Model

- Pretrained ConvNeXtXLarge model with frozen base
- Custom top layers with batch norm and dropout
- Adam optimizer (LR 1e-2), early stopping, class weights
- Online test accuracy: 38%.

### 4️⃣ Hybrid Model: EfficientNetB0 + VGG16

- Both used as frozen feature extractors
- GlobalAveragePooling2D + feature concatenation + custom dense head
- Some convolutional layers unfrozen for fine-tuning
- Online test accuracy: 49% (best result in this project).

---

## Results

| Metric                | Value |
|-----------------------|-------|
| Local test accuracy   | 0.98  |
| Hidden test set accuracy (best) | 0.52  |

- Data cleaning, augmentation, class weighting all contributed to improvements.
- Overfitting remained a challenge despite regularization and augmentation.
- Transfer learning with large pretrained models had limited gains.

---

## Observations

### Strengths

- Extensive data preprocessing and augmentation improved generalization.
- Class weighting improved fairness across classes.
- Hybrid architecture (EfficientNetB0 + VGG16) leveraged strengths of both networks.

### Weaknesses

- Overfitting to local dataset remained an issue.
- Pretrained models (DenseNet, ConvNeXt) did not fully transfer well to the blood cell domain.
- Hybrid model increased computational cost with modest gains.

### Limitations

- Some augmentations may have distorted important cell features.
- Limited performance gain from large pretrained models due to domain mismatch.

---

## Suggested Improvements & Future Work

- Stronger regularization: higher dropout, L1/L2 penalties, improved early stopping.
- Ensemble learning: combine multiple model predictions to improve robustness.
- Targeted augmentation: simulate microscopy artifacts to enhance generalization.
- Explore advanced architectures (Vision Transformers, Swin Transformers).
- Apply domain adaptation techniques to bridge gap between ImageNet pretraining and blood cell domain.

---

> 📌 This project demonstrates my ability to design and optimize deep learning models for **challenging small medical datasets**, effectively using **transfer learning, augmentation, and hybrid architectures** with PyTorch/Keras.


