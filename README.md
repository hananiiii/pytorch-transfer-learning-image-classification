# PyTorch Transfer Learning for Image Classification

This repository contains a PyTorch-based project demonstrating **transfer learning** to solve an image classification problem. Compared to training a CNN from scratch (like TinyVGG), transfer learning significantly improves model accuracy and reduces loss.

---

##  Project Overview

In this notebook:
- We compare **TinyVGG (custom CNN)** with **EfficientNetB0 (pre-trained)**.
- Use PyTorch's `torchvision.models` for loading pre-trained models.
- Apply **transforms**, **normalization**, and **data loaders**.
- Implement **transfer learning** by freezing base layers and training the classifier head.
- Evaluate performance using **accuracy**, **loss**, and visualizations.

---

##  Key Concepts

-  **Transfer Learning**: Reusing a pre-trained model (e.g., EfficientNet) to improve accuracy on a new dataset.
-  **Model Comparison**: Shows how increasing model complexity improves performance.
-  **Transforms**: Normalize input using `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]` to match pretrained model expectations.
-  **NUM_WORKERS**: Used in the `DataLoader` for parallel data loading.

---

##  Model Details

| Model          | Parameters     | Accuracy | Loss   |
|----------------|----------------|----------|--------|
| TinyVGG        | ~8K            | Low      | High   |
| EfficientNetB0 | ~5.2 Million   | High     | Low    |

## References
PyTorch Transfer Learning Tutorial

EfficientNet PyTorch


