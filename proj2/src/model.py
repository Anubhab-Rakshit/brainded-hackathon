import torch
import torch.nn as nn
from .modules.pro_fa import HierarchicalVisualEncoder
from .modules.mix_mlp import MixMLPClassifier
from .modules.rcta import CognitiveDecoder

class CognitiveRadiologyModel(nn.Module):
    def __init__(self, config):
        super(CognitiveRadiologyModel, self).__init__()
        self.config = config
        
        # Module 1: PRO-FA
        self.visual_encoder = HierarchicalVisualEncoder(
            model_name=config.VIT_MODEL_NAME
        )
        
        # Module 2: MIX-MLP
        self.classifier = MixMLPClassifier(
            input_dim=config.VISUAL_EMBED_DIM,
            num_diseases=config.NUM_DISEASES,
            hidden_dim=config.CLASSIFIER_HIDDEN_DIM
        )
        
        # Module 3: RCTA
        self.decoder = CognitiveDecoder(
            vocab_size=config.VOCAB_SIZE,
            embed_dim=config.DECODER_DIM,
            num_layers=config.DECODER_LAYERS,
            num_heads=config.DECODER_HEADS,
            max_len=config.MAX_SEQ_LENGTH
        )
        
        # Bindings / Projections if dims mismatch
        if config.VISUAL_EMBED_DIM != config.DECODER_DIM:
            self.visual_proj = nn.Linear(config.VISUAL_EMBED_DIM, config.DECODER_DIM)
        else:
            self.visual_proj = nn.Identity()
            
        # Label Embeddings
        # We need to embed the predicted labels for the RCTA module
        self.label_embedding = nn.Embedding(config.NUM_DISEASES, config.DECODER_DIM)
        
    def forward(self, images, input_ids=None, labels=None):
        # images: [B, 3, 224, 224]
        # input_ids: [B, T] (Target report for training)
        # labels: [B, Num_Diseases] (Ground truth disease labels for training Module 2)
        
        # 1. Visual Features (PRO-FA)
        visual_features = self.visual_encoder(images)
        pixel_feat = visual_features['pixel']
        region_feat = visual_features['region']
        organ_feat = visual_features['organ']
        
        # 2. Disease Classification (MIX-MLP)
        cls_logits, enhanced_feat = self.classifier(organ_feat)
        
        # 3. Report Generation (RCTA)
        # Prepare memories for decoder
        
        # Project visual features to decoder dim
        image_memory = self.visual_proj(pixel_feat) # [B, 196, D]
        
        # Prepare Label Memory
        # For training, we can use GT labels or predicted logits.
        # "Context queries Predicted Labels" -> We should use the model's prediction context.
        # We can embed the disease indices weighted by checking top-k or just embedding all with weights.
        # Simplification: Embed the Disease ID, weight by sigmoid(logits).
        
        # Create a weighted embedding of diseases
        # [B, Num_Diseases] -> [B, Num_Diseases, D]
        disease_weights = torch.sigmoid(cls_logits).unsqueeze(-1) # [B, 14, 1]
        all_disease_ids = torch.arange(self.config.NUM_DISEASES, device=images.device).unsqueeze(0).repeat(images.shape[0], 1)
        disease_embeds = self.label_embedding(all_disease_ids) # [B, 14, D]
        label_memory = disease_embeds * disease_weights # Weighted embeddings
        
        # Text Memory (Clinical Indication)
        # In this simplified pipeline, we might not have separate indication text input.
        # If not available, we can use the "enhanced_feat" from MIX-MLP as a global context proxy.
        text_memory = self.visual_proj(enhanced_feat).unsqueeze(1) # [B, 1, D]
        
        decoder_logits = None
        if input_ids is not None:
            # Training Mode
            decoder_logits = self.decoder(
                input_ids=input_ids,
                text_memory=text_memory,
                label_memory=label_memory,
                image_memory=image_memory
            )
            
        return {
            'cls_logits': cls_logits,
            'decoder_logits': decoder_logits
        }
