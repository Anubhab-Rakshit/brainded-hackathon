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
  ## Fig1: Genre Popularity of Different Movies
    ![GENRE POPULATION](https://github.com/user-attachments/assets/16163859-9823-4945-93ea-25834a955d16)
  ## Fig 2: WorkFlow Architecture of Reel Sense
  ![workflow reelsense](https://github.com/user-attachments/assets/84af5dc0-e6c1-48ea-9798-66785fba3c11)
  ## Fig 3: Long Tail Distribution vs No.of Rating Graph Based on Movie Rank
  ![Fig 3 Long Tail Distribution vs No of Rating Graph Based on Movie Rank](https://github.com/user-attachments/assets/aec361c5-fd90-4244-9481-a63d83decc82)
  ## Fig 4: Distribution of Movies Rating Per Count
  ![Fig 4: Distribution of Movies Rating Per Count](https://github.com/user-attachments/assets/1fb53d6b-51c9-4a72-b501-78ed2642eaf5)



  


📂 **Code:** Located in `proj1/` directory.

📂 **Report** Located in `report/` directory.

---

## 2. Cognitive Radiology (Project 2)
**AI-Powered Medical Report Generation**

A "Second Reader" for radiologists. This deep learning system analyzes Chest X-Rays and automatically generates professional textual radiology reports (Findings + Impression). It uses a Hierarchical Visual Transformer (PRO-FA) and a Knowledge-Enhanced Decoder (RCTA) to detect pathologies like Pneumonia, Cardiomegaly, and Effusion.

- **Key Features:**
  - **Vision Transformer (ViT):** High-performance visual feature extraction.
  - **RCTA Decoder:** Relational Cross-Attention for clinically accurate text generation.
  - **Pathology Detection:** Classifies 14 distinct chest diseases with high sensitivity.
  - **Demo Ready:** Includes inference scripts for real-time analysis.
## Workflow of Cognative Radiology:
<img width="678" height="777" alt="image" src="https://github.com/user-attachments/assets/d3381334-ae55-47fa-aa39-a00830d25583" />

## Fig: Estimate Representation of Cognitive Radiology:
<img width="727" height="460" alt="Screenshot 2026-02-08 203801" src="https://github.com/user-attachments/assets/f31e90db-3212-45c8-b44a-c2c3dcb633c7" />



📂 **Code:** Located in `proj2/` directory.

📂 **Report** Located in `report/` directory.

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
*Developed by 2 Bit Engineers*
