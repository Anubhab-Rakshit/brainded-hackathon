import torch
import torch.nn as nn
import timm

class HierarchicalVisualEncoder(nn.Module):
    """
    PRO-FA Module: Hierarchical Visual Perception.
    Extracts visual features at three granularities:
    1. Pixel-level (local patches)
    2. Region-level (mid-level semantic regions)
    3. Organ-level (global representation)
    """
    def __init__(self, model_name='vit_base_patch16_224', pretrained=False):
        super(HierarchicalVisualEncoder, self).__init__()
        # Load pre-trained ViT from timm
        # We need to access intermediate layers, so we might need to hook or use features_only if available
        # But ViT is a flat hierarchy. We can extract features from different depths.
        
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.hidden_dim = self.vit.embed_dim
        
        # Define layers to extract from for "Region" level
        # For ViT-Base (12 layers), let's use layers 6 and 9 as intermediate region representations
        # But standard timm vit forward doesn't return intermediate easily unless we modify it or use forward_features
        # A simpler approximation for "Region" in ViT is the patch embeddings themselves at the final layer,
        # tailored by an attention pooling or similar, but the prompt says 
        # "Your visual encoder (e.g., ViT) must extract features at three distinct granularities"
        
        # 1. Pixel Level: Patch embeddings from the last layer (minus CLS)
        # 2. Organ Level: The CLS token from the last layer.
        # 3. Region Level: We can simulate this by pooling patches or taking an intermediate layer.
        # Let's use an intermediate layer for "Region" to capture lower-level semantics.
        
    def forward(self, x):
        # x: [B, 3, 224, 224]
        
        # We need to run the ViT and capture intermediate states.
        # Timm's `forward_features` usually returns the final layer features (CLS + patches)
        
        # Let's assume standard ViT behavior:
        # x = self.vit.patch_embed(x)
        # x = self.vit._pos_embed(x)
        # x = self.vit.blocks(x) 
        
        # For a hackathon, let's stick to the high-level features from the final layer 
        # but map them conceptually:
        # - Pixel: Individual patches (N_patches, Dim)
        # - Organ: CLS token (1, Dim)
        # - Region: We can use AdaptiveAvgPool on patches to get coarse grids, e.g. 7x7 -> 3x3
        
        features = self.vit.forward_features(x) # [B, 197, 768] for ViT-B/16 (196 patches + 1 CLS)
        
        # Extract CLS and Patches
        organ_level = features[:, 0, :] # [B, 768] (CLS)
        pixel_level = features[:, 1:, :] # [B, 196, 768] (Patches)
        
        # Generate Region Level by pooling pixel level
        # Reshape to spatial grid [B, 14, 14, 768]
        B, N, C = pixel_level.shape
        H = W = int(N**0.5) # 14 for 224x224
        
        reshaped_pixels = pixel_level.permute(0, 2, 1).view(B, C, H, W)
        
        # Adaptive pool to create "Regions" (e.g., 2x2 or 3x3 grid of regions)
        # Fix for MPS Runtime Error: "Adaptive pool MPS: input sizes must be divisible by output sizes"
        # We temporarily move to CPU for this operation if on MPS
        original_device = reshaped_pixels.device
        if original_device.type == 'mps':
            reshaped_pixels = reshaped_pixels.cpu()
            
        region_level_pooled = nn.functional.adaptive_avg_pool2d(reshaped_pixels, (3, 3)) # [B, C, 3, 3]
        
        if original_device.type == 'mps':
            region_level_pooled = region_level_pooled.to(original_device)
        region_level = region_level_pooled.flatten(2).transpose(1, 2) # [B, 9, 768]
        
        return {
            "pixel": pixel_level,   # [B, 196, 768]
            "region": region_level, # [B, 9, 768]
            "organ": organ_level    # [B, 768]
        }
