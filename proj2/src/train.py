import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from ..config import config
from ..src.data.dataset import MIMICCXRDataset, MockRadiologyDataset, get_transforms, get_tokenizer
from ..src.model import CognitiveRadiologyModel

def train():
    display_device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
        display_device = "cuda"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        display_device = "mps (Apple Silicon Acceleration)"
    else:
        device = torch.device("cpu")
        
    # FINAL OPTIMIZATION: Enable TF32 for RTX 5090 (Free speedup on FP32 ops)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"Using device: {display_device}")
    
    # tokenizer
    tokenizer = get_tokenizer()
    # Update config vocab size
    config.VOCAB_SIZE = tokenizer.vocab_size
    
    # Download Datasets
    config.download_datasets()
    
    # Dataset & DataLoader
    if config.MIMIC_CXR_ROOT and os.path.exists(config.MIMIC_CXR_ROOT):
        print(f"Using MIMIC-CXR from {config.MIMIC_CXR_ROOT}")
        train_dataset = MIMICCXRDataset(
            root_dir=config.MIMIC_CXR_ROOT,
            csv_file=os.path.join(config.MIMIC_CXR_ROOT, "mimic_cxr_aug_train.csv"),
            tokenizer=tokenizer,
            transforms=get_transforms(is_train=True),
            max_samples=config.MAX_SAMPLES 
        )
    else:
        print("MIMIC-CXR not found. Falling back to MockDataset.")
        train_dataset = MockRadiologyDataset(
            tokenizer=tokenizer, 
            transforms=get_transforms(is_train=True),
            num_samples=100
        )
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    
    # Model
    model = CognitiveRadiologyModel(config).to(device)
    
    # Optimizers (SOTA Recipe for ViT: Weight Decay 0.05, Betas 0.9/0.95)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.05, betas=(0.9, 0.95))
    
    # Loss Functions (Label Smoothing for Better Generalization)
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_gen = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
    
    # Scheduler (Cosine with Warmup)
    from transformers import get_cosine_schedule_with_warmup
    total_steps = len(train_loader) * config.NUM_EPOCHS
    warmup_steps = int(0.1 * total_steps) 
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Training Loop
    model.train()
    for epoch in range(config.NUM_EPOCHS):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        total_loss = 0
        
        for batch in loop:
            images = batch['image'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                decoder_input = input_ids[:, :-1]
                decoder_target = input_ids[:, 1:]
                
                outputs = model(images, input_ids=decoder_input, labels=labels)
                
                cls_logits = outputs['cls_logits']
                decoder_logits = outputs['decoder_logits'] 
                
                loss_cls = criterion_cls(cls_logits, labels)
                
                loss_gen = criterion_gen(
                    decoder_logits.reshape(-1, config.VOCAB_SIZE), 
                    decoder_target.reshape(-1)
                )
                
                loss = loss_cls + loss_gen
            
            # Backward
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step() 
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss}")
        
        # Save Checkpoint EVERY EPOCH (Safety First)
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    train()
