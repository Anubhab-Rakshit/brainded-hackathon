# BrainDed Hackathon Project

Welcome to the **BrainDed** repository. This monorepo contains two cutting-edge AI projects developed for the hackathon:

## 1. ReelSense (Project 1)
**Advanced Movie Recommender System**

ReelSense is a state-of-the-art recommendation engine that goes beyond simple collaborative filtering. It leverages a hybrid architecture combining matrix factorization, deep learning (Neural Collaborative Filtering), and graph neural networks (LightGCN) to provide highly personalized content suggestions.

- **Key Features:**
  - **Hybrid Engine:** Combines SVD, NCF, and LightGCN.
  - **Explainable AI:** Provides tag-based explanations for every recommendation.
  - **Diversity Optimization:** Ensuring users aren't trapped in filter bubbles.
  - **Real-Time Pipeline:** Efficient data loaders and inference scripts.

📂 **Code:** Located in `proj1/` directory.

---

## 2. Cognitive Radiology (Project 2)
**AI-Powered Medical Report Generation**

A "Second Reader" for radiologists. This deep learning system analyzes Chest X-Rays and automatically generates professional textual radiology reports (Findings + Impression). It uses a Hierarchical Visual Transformer (PRO-FA) and a Knowledge-Enhanced Decoder (RCTA) to detect pathologies like Pneumonia, Cardiomegaly, and Effusion.

- **Key Features:**
  - **Vision Transformer (ViT):** High-performance visual feature extraction.
  - **RCTA Decoder:** Relational Cross-Attention for clinically accurate text generation.
  - **Pathology Detection:** Classifies 14 distinct chest diseases with high sensitivity.
  - **Demo Ready:** Includes inference scripts for real-time analysis.

📂 **Code:** Located in `proj2/` directory.

---

## Getting Started

### Prerequisites
- Python 3.10+
- PyTorch (CUDA/MPS supported)

### Installation
```bash
pip install -r requirements.txt
```

### Running the Demos

**ReelSense:**
```bash
cd proj1
python main_pipeline.py
```

**Cognitive Radiology:**
```bash
# Ensure model is downloaded to proj2/model_final.pth
python -m proj2.src.inference --image test_image.jpg --checkpoint proj2/model_final.pth
```

---
*Developed by Anubhab Rakshit*
