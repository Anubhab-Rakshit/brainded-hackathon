
import torch
import os
import argparse
from PIL import Image
from transformers import BertTokenizer
import matplotlib.pyplot as plt

from proj2.config import config
from proj2.src.model import CognitiveRadiologyModel
from proj2.src.data.dataset import get_transforms

def load_model(checkpoint_path, device):
    """
    Loads the trained Cognitive Radiology Model from a checkpoint.
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Initialize Model Structure
    model = CognitiveRadiologyModel(config)
    
    # Load Weights
    # map_location ensures we can load CUDA weights onto CPU/MPS
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model

def generate_report(model, image_path, tokenizer, device, max_len=100):
    """
    Generates a medical report for a given chest X-ray image.
    """
    # 1. Preprocess Image
    transform = get_transforms(is_train=False)
    image = Image.open(image_path).convert("RGB")
    
    # Optional: Show image
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()
    
    image_tensor = transform(image).unsqueeze(0).to(device) # [1, 3, 224, 224]
    
    # 2. visual Encoder Forward
    with torch.no_grad():
        visual_features = model.visual_encoder(image_tensor)
        pixel_feat = visual_features['pixel']
        region_feat = visual_features['region']
        organ_feat = visual_features['organ']
        
        # Classifier
        cls_logits, enhanced_feat = model.classifier(organ_feat)
        
        # Get Disease Tags
        disease_probs = torch.sigmoid(cls_logits)
        # We can print these as "Detected Anomalies"
        
        # Prepare Decoder Memory
        image_memory = model.visual_proj(pixel_feat)
        
        # Label Memory (Weighted by predictions)
        disease_weights = disease_probs.unsqueeze(-1)
        all_disease_ids = torch.arange(config.NUM_DISEASES, device=device).unsqueeze(0)
        disease_embeds = model.label_embedding(all_disease_ids)
        label_memory = disease_embeds * disease_weights
        
        # Text Memory (Proxy)
        text_memory = model.visual_proj(enhanced_feat).unsqueeze(1)
        
        # 3. Greedy Decoding
        # Start Token
        start_token = tokenizer.cls_token_id
        end_token = tokenizer.sep_token_id
        
        generated_ids = [start_token]
        
        for _ in range(max_len):
            input_seq = torch.tensor([generated_ids], device=device)
            
            # Decoder Forward
            # We need to expose the caching mechanism or re-run forward for simplicity
            # For this simple implementation, we assume the decoder can handle the full sequence so far
            # RCTA decoder expects `input_ids` as simple tensor
            
            logits = model.decoder(
                input_ids=input_seq,
                text_memory=text_memory,
                label_memory=label_memory,
                image_memory=image_memory
            )
            
            # Get last token logits
            next_token_logits = logits[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1).item()
            
            if next_token_id == end_token:
                break
                
            generated_ids.append(next_token_id)
            
    # Decode to Text
    report = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return report, disease_probs

def main():
    parser = argparse.ArgumentParser(description="Cognitive Radiology Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to chest x-ray image")
    parser.add_argument("--checkpoint", type=str, default="model_epoch_20.pth", help="Path to trained .pth model")
    args = parser.parse_args()
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Running inference on {device}")
    
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Load Model
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        print("Please download it from the server first.")
        return
        
    model = load_model(args.checkpoint, device)
    
    # Generate
    print(f"Generating report for: {args.image}...")
    report, diseases = generate_report(model, args.image, tokenizer, device)
    
    print("\n" + "="*40)
    print("   RADIOLOGY REPORT (Generated)   ")
    print("="*40)
    print(report)
    print("="*40)
    
    print("\nDetected Pathology Probabilities:")
    # Assuming standard order (ChestXRay-14 / MIMIC labels)
    # Ideally should load label map. 
    # For now, just print raw vector or top findings if we had labels.
    print(diseases.cpu().numpy().round(2))

if __name__ == "__main__":
    main()
