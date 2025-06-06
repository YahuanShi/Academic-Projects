# Artificial Neural Networks and Deep Learning

## 🎯 Course Description

This course provides a broad and rigorous introduction to neural networks and modern deep learning techniques. It covers theoretical foundations, training algorithms, and hands-on implementation of deep architectures. The focus is both on understanding core principles and applying deep learning models to solve complex engineering problems.

Topics include:
- Feedforward neural networks and backpropagation
- Regularization techniques: weight decay, dropout, early stopping
- Convolutional Neural Networks (CNNs) and their variants
- Recurrent Neural Networks (RNNs), LSTMs, and sequence modeling
- Attention mechanisms, Transformers, and sequence-to-sequence learning
- Autoencoders, GANs, and representation learning
- Application of deep learning to computer vision tasks

---

## 📘 Learning Outcomes

After completing this course, I was able to:

- Understand theoretical results underpinning neural networks (universal approximation, vanishing/exploding gradients)
- Train and evaluate deep models using optimization techniques (SGD, Adam, etc.)
- Implement and fine-tune CNN, RNN, and Transformer-based architectures
- Apply deep learning models to real-world tasks such as classification, segmentation, and object detection
- Use PyTorch to develop, train, and visualize deep models

---

| Assignment | Folder | Description |
|------------|--------|-------------|
| Assignment 1 | [`assignment-01`](./assignment-01) | **Blood Cell Image Classification**. Built and optimized CNN-based and hybrid architectures to classify 96x96 RGB images of 8 blood cell types. Applied data cleaning, augmentation, class weighting, and transfer learning (DenseNet169, ConvNeXtXLarge, EfficientNetB0 + VGG16 hybrid). Evaluated with accuracy and confusion matrix on both local and hidden test sets. |
| Assignment 2 | [`assignment-02`](./assignment-02) | **Mars Terrain Semantic Segmentation**. Designed and trained a U-Net-based CNN with residual connections for pixel-wise segmentation of Mars terrain images into 5 classes. Addressed severe class imbalance using weighted loss and oversampling. Evaluated with Mean IoU (excluding background), achieving strong generalization to hidden test set. |

---

## 🛠️ Tools & Libraries

- Python 3
- PyTorch
- NumPy, pandas
- Matplotlib, seaborn
- scikit-learn
- Jupyter Notebooks

---

## 📚 Teaching Materials

- **Book**: *Deep Learning* by Ian Goodfellow, Yoshua Bengio, Aaron Courville (MIT Press)
- **Slides**: From Matteo Matteucci and Giacomo Boracchi (available via course link)
- **Practicals**: Hands-on lab sessions with annotated notebooks (CNN, RNN, Autoencoders)

---

## 📝 Notes

The course combined both theoretical depth and practical coding exercises. Assignments were modeled as Kaggle challenges, testing both my modeling and problem-solving skills. The final projects demonstrate a complete pipeline: data preparation, model design, training, evaluation, and visualization.

> 📂 Browse the assignment folders for full code, experiments, and reports.
