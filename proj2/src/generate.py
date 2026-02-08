import torch
from PIL import Image
from ..config import config
from ..src.model import CognitiveRadiologyModel
from ..src.data.dataset import get_transforms, get_tokenizer

def generate_report(image_path, model_path, device='cpu'):
    # Load Model
    model = CognitiveRadiologyModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load Image
    transforms = get_transforms(is_train=False)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transforms(image).unsqueeze(0).to(device)
    
    # Text Tokenizer
    tokenizer = get_tokenizer()
    
    # Greedy Decoding
    # Start with [BOS]
    # We need to know BOS token id. AutoTokenizer usually has cls_token or bos_token
    bos_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    if bos_token_id is None: bos_token_id = 101 # Fallback for BERT
    
    decoder_input = torch.tensor([[bos_token_id]], device=device)
    
    generated_tokens = []
    
    with torch.no_grad():
        # Get Visual Features & Class Consensus first
        # We need to expose a method in model to cache encoder outputs to avoid re-computing
        # For simplicity calling forward repeatedly or we modify model.py to separate encoder/decoder
        # Let's modify the loop to validly use the forward pass step-by-step
        
        # 1. Encode Image
        visual_features = model.visual_encoder(image_tensor)
        cls_logits, enhanced_feat = model.classifier(visual_features['organ'])
        
        # Prepare memories
        if config.VISUAL_EMBED_DIM != config.DECODER_DIM:
            pixel_feat = model.visual_proj(visual_features['pixel'])
            text_memory = model.visual_proj(enhanced_feat).unsqueeze(1)
        else:
            pixel_feat = visual_features['pixel']
            text_memory = enhanced_feat.unsqueeze(1)
            
        # Label Memory
        disease_weights = torch.sigmoid(cls_logits).unsqueeze(-1)
        all_disease_ids = torch.arange(config.NUM_DISEASES, device=device).unsqueeze(0)
        disease_embeds = model.label_embedding(all_disease_ids)
        label_memory = disease_embeds * disease_weights
        
        # 2. Decode Loop
        for _ in range(config.MAX_SEQ_LENGTH):
            logits = model.decoder(
                input_ids=decoder_input,
                text_memory=text_memory,
                label_memory=label_memory,
                image_memory=pixel_feat
            )
            
            # Get last token logits
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            
            if next_token_id == tokenizer.sep_token_id or next_token_id == tokenizer.eos_token_id:
                break
                
            generated_tokens.append(next_token_id)
            decoder_input = torch.cat([decoder_input, torch.tensor([[next_token_id]], device=device)], dim=1)
            
    decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return decoded_text

if __name__ == "__main__":
    # Example usage
    # report = generate_report("test.jpg", "checkpoints/model_epoch_20.pth")
    # print(report)
    pass
