# Cognitive Radiology: Workflow Diagrams

Use these diagrams for your presentation slides.

## 1. High-Level System Architecture

```mermaid
graph TD
    Input[Chest X-Ray Image] --> ViT[ViT-Large Backbone]
    ViT -->|Extract features| PROFA[PRO-FA Module]
    
    subgraph "Visual Processing"
        PROFA -->|Fine details| Pixel[Pixel Features]
        PROFA -->|Anatomy| Region[Region Features]
        PROFA -->|Global| Organ[Organ Features]
    end
    
    Organ --> MIXMLP[MIX-MLP Classifier]
    MIXMLP -->|Predicts| DiseaseLabels[Disease Probabilities]
    
    subgraph "Generation (RCTA Decoder)"
        Pixel --> CrossAttn1[Visual Attention]
        DiseaseLabels --> CrossAttn2[Label Attention]
        CrossAttn1 --> Decoder[Transformer Decoder]
        CrossAttn2 --> Decoder
    end
    
    Decoder --> Output[Final Medical Report]
    
    style ViT fill:#f9f,stroke:#333
    style PROFA fill:#bbf,stroke:#333
    style MIXMLP fill:#bfb,stroke:#333
    style Decoder fill:#fbb,stroke:#333
```

## 2. High-Performance Training Pipeline (RTX 5090)

```mermaid
sequenceDiagram
    participant Cloud as Cloud Server (Vast.ai)
    participant Data as MIMIC-CXR Database
    participant CPU as System CPU (8 Workers)
    participant GPU as RTX 5090 (24GB VRAM)
    
    Cloud->>Data: Authenticate (kaggle.json)
    Data->>Cloud: Download 40GB Dataset (10Gbps)
    
    loop Training Epoch
        CPU->>CPU: Load & Augment Batch (64 Images)
        CPU->>GPU: Pin Memory & Transfer
        
        activate GPU
        GPU->>GPU: Forward Pass (TF32 Precision)
        GPU->>GPU: Calculate Loss (Label Smoothing)
        GPU->>GPU: Backward Pass (AMP Scaled)
        GPU->>GPU: Optimizer Step (AdamW)
        deactivate GPU
        
        GPU-->>Cloud: Log Metrics (Loss, Accuracy)
    end
    
    Cloud->>Cloud: Save Checkpoint
```

## 3. Inference Logic (How it "Thinks")

```mermaid
flowchart LR
    img((Image)) -->|See| Vision[Visual Encoder]
    Vision -->|Diagnose| Dx[Detected: Cardiomegaly]
    
    Dx -->|Reason| Reasoner{RCTA Logic}
    Vision -->|Reason| Reasoner
    
    Reasoner -->|Write| T1[Token 1: 'The']
    Reasoner -->|Write| T2[Token 2: 'heart']
    Reasoner -->|Write| T3[Token 3: 'is']
    Reasoner -->|Write| T4[Token 4: 'enlarged']
    
    T1 & T2 & T3 & T4 --> Report[Final Report]
```
