# Assignment 2 — Image Segmentation of Mars Dataset

## 🧠 Task Overview

This assignment focuses on solving an **image segmentation** task using deep neural networks. The goal is to accurately segment **Mars terrain images** into 5 terrain classes at pixel level.

---

## 🚀 Dataset Description

- **Dataset file**: `mars_for_students.npz`
- **Image size**: 64 x 128 (grayscale)
- **Training set shape**: `(samples, 64, 128, 1)`
- **Target masks**: pixel-wise label masks with 5 classes
- **Number of Classes**:

| Class | Label |
|-------|------------------------------|
| 0     | Background                   |
| 1     | Soil                         |
| 2     | Bedrock                      |
| 3     | Sand                         |
| 4     | Big Rock                     |

- **Challenges**:
  - Severe class imbalance (background dominates)
  - Generalization to unseen test data
  - Computational efficiency constraints

---

## 🛠️ Approach

### 🔹 Data Preparation

- Normalized input images to `[0, 1]`
- Converted labels to one-hot encoded masks
- Removed outliers ("alien" samples), with minor impact
- Oversampled Class 4 (Big Rock) to mitigate imbalance
- Implemented class weights to balance loss contributions

### 🔹 Model Architecture

- **U-Net-based encoder-decoder architecture**:
  - Symmetric encoder-decoder with skip connections
  - Residual connections added for stability
  - Dropout and L2 regularization applied to combat overfitting

- **Loss Function**:
  - Weighted categorical cross-entropy
  - Class weights computed based on pixel counts

- **Evaluation Metric**:
  - Custom Mean IoU, excluding Background (Class 0)

### 🔹 Training Settings

- Optimizer: AdamW
- Learning Rate: initial 1e-2, reduced on plateau to minimum 1e-6
- Scheduler: Halves LR upon validation stagnation
- EarlyStopping to prevent overfitting

- Batch Size:
  - Version 1: 124 (standard U-Net)
  - Version 2: 32 (regularized model, more stable)

---

## 📊 Results

| Metric                | Value |
|-----------------------|-------|
| Training Accuracy     | 0.92 |
| Validation Accuracy   | 0.63  |
| Hidden Test Set Accuracy | 0.61 |

- Best model uses **regularized U-Net** with dropout, L2, and class balancing
- Advanced loss functions (Dice Loss, Focal Loss) were tested but not superior to weighted cross-entropy
- Deeper networks led to overfitting, performance worsened

### 📈 Training & Validation Curves

- Regularization improved stability of validation Mean IoU
- Reduced gap between training and validation loss/metrics
- Final model demonstrated better generalization than baseline

---

## 📝 Observations

- **Class imbalance** was a major challenge; class weighting and oversampling helped.
- **Dropout and L2 regularization** effectively reduced overfitting.
- **Learning rate scheduler** improved training stability.
- Increasing model depth led to overfitting due to limited dataset size.
- The chosen U-Net variant with residual connections achieved the best tradeoff between accuracy and stability.

---

---

## 🧪 Future Improvements

- **Loss function refinement**: dynamic re-weighting strategies
- **Advanced architectures**: attention mechanisms, hybrid models
- **Data augmentation**: more aggressive augmentation to improve generalization
- **Training stability**: further tuning LR schedules and regularization

---

> 📌 This project demonstrates my ability to design and implement **CNN-based segmentation models** for **challenging real-world datasets**, with careful attention to **class imbalance, regularization**, and **training optimization** using TensorFlow and Keras.

