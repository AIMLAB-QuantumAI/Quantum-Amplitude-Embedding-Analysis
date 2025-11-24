# Quantum-Amplitude-Embedding-Analysis

> **Quantum Fine-tuning Framework with CNN + Quantum Embedding Layer (QEM) + LoRA Extension**  
> This repository explores how *Quantum Amplitude Embedding* compares with *Classical LoRA-based fine-tuning*  
> in terms of convergence, gradient stability, and representational power.

---

## Overview

This project investigates how **Quantum Embedding Layers (QEM)** can be integrated with a **pretrained CNN backbone**  
and compared against **LoRA (Low-Rank Adaptation)** fine-tuning.

We systematically track:
- Gradient constancy and stability (`grad_norm`, `grad_const`, `grad_nan`)
- Loss convergence trends
- PCA / t-SNE visualization of embedding representations

---

## Model Architectures

### CNN Backbone
```python
class CNN(nn.Module):
    def __init__(self, hidden_dim=1024, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

## 251124 Version
