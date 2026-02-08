import torch
import torch.nn as nn
import torch.nn.functional as F

class MixMLPClassifier(nn.Module):
    """
    MIX-MLP: Knowledge-Enhanced Classification.
    Predicts disease tags (14 CheXpert labels) from the global visual features.
    Uses a dual-path architecture:
    1. Expansion Path: Projects to high-dim space to capture complex correlations.
    2. Residual Path: Preserves original signal.
    """
    def __init__(self, input_dim=768, num_diseases=14, hidden_dim=2048, dropout=0.1):
        super(MixMLPClassifier, self).__init__()
        
        # Expansion Path
        self.expansion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_diseases)
        )
        
    def forward(self, x):
        # x: [B, input_dim] (Organ-level feature)
        
        # Residual Connection
        residual = x
        
        # Expansion Path
        out = self.expansion(x)
        
        # Combine
        x = residual + out
        
        # Classification
        logits = self.classifier(x)
        
        return logits, x # Return logits and the enhanced feature vector
