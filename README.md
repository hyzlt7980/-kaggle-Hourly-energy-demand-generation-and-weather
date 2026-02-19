# XManifoldUltra: Advanced Energy Price Forecasting

**XManifoldUltra** is a high-performance deep learning architecture designed for binary classification of energy price movements (predicting if the price will go up or down). By integrating a **Five-Expert Gating System (MoE)** with a **Transformer-based inter-variable encoder**, this model is engineered to capture complex patterns in volatile time-series data while maintaining a strict "zero-leakage" training environment.

---

## 🏗️ Model Architecture

The model processes multi-variate time-series data through several sophisticated layers to ensure robust feature extraction and relationship modeling.

![XManifoldUltra Architecture](xmanifold.png)

### Core Components
* **RevIN (Reversible Instance Normalization):** Addresses the non-stationary nature of energy prices by normalizing distributions and handling distribution shift.
* **IntraColumnPentaExperts:** A five-branch Mixture of Experts (MoE) system that analyzes each feature through specialized lenses:
    1.  **Global Expert:** Captures long-term trends via linear projection.
    2.  **Local Expert:** Identifies short-term patterns using 1D-CNNs.
    3.  **Diff Expert:** Focuses on the rate of change by processing first-order derivatives (price deltas).
    4.  **Swin Expert:** Utilizes a **1D Swin Transformer** with shifted windows for hierarchical temporal attention.
    5.  **Sliding Expert:** Uses convolutional windows and adaptive pooling for localized feature capture.
* **Game Room (Transformer Encoder):** A 3-layer Transformer that models the complex, "game-theoretic" relationships between different variables (e.g., how solar generation interacts with demand to influence price).
* **Zero-Leakage Dataset:** A rigorous data pipeline that enforces a 1-step lag on all features, ensuring the model never "sees" the future during training.

---

## 🚀 Performance & Optimization

During testing on the Spain Energy Dataset, the model demonstrated high stability and predictive power:
* **Validation Accuracy:** Achieved a peak validation accuracy of **~82.09%**.
* **Loss Function:** Utilizes **Smoothed BCE Loss** (smoothing factor of 0.40) to prevent overfitting on noisy financial labels.
* **Training Infrastructure:** Optimized for multi-GPU setups using **Distributed Data Parallel (DDP)** and **Mixed Precision (FP16)** via GradScaler.

---

## 💻 Setup & Training

### Requirements
* PyTorch 2.x
* NVIDIA GPUs (Multi-GPU recommended for DDP)
* `pandas`, `numpy`, `tqdm`

### Distributed Training Command
To launch training across 2 GPUs using the `torchrun` utility:
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 main_script.py