import os
import kagglehub

class Config:
    # --- Data Paths ---
    # We will set these dynamically if using kagglehub, or user can override
    MIMIC_CXR_ROOT = None 
    IU_XRAY_ROOT = None
    
    # Checkpoints and Outputs
    OUTPUT_DIR = "./output/cognitive_radiology"
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    
    # --- Model Hyperparameters ---
    # Visual Encoder (PRO-FA)
    # Beast Mode for RTX 5090 (24GB+ VRAM)
    VIT_MODEL_NAME = "vit_large_patch16_224" # Upgrade to ViT-Large
    VISUAL_EMBED_DIM = 1024 # Matches ViT-Large
    
    # Classifier (MIX-MLP)
    NUM_DISEASES = 14
    CLASSIFIER_HIDDEN_DIM = 4096 # Double hidden dim
    
    # Decoder (RCTA)
    VOCAB_SIZE = 30522 
    MAX_SEQ_LENGTH = 128 # Slightly longer sequences
    DECODER_LAYERS = 8 # Deeper decoder
    DECODER_HEADS = 16 # More attention heads
    DECODER_DIM = 1024 # Match visual embed dim
    DROPOUT = 0.1
    
    # --- Training ---
    # 5090 can handle huge batches. This stabilizes training and speeds it up.
    # 5090 can handle huge batches. This stabilizes training and speeds it up.
    BATCH_SIZE = 64 
    LEARNING_RATE = 5e-5 # Slightly higher LR for larger batch 
    NUM_EPOCHS = 20
    MAX_SAMPLES = None # GOD MODE: Train on the ENTIRE dataset (370k+ images)
    
    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

    def download_datasets(self):
        print("Downloading datasets via KaggleHub... This may take a while.")
        try:
            # IU X-Ray (Smaller, ~2GB?)
            print("Downloading IU-Xray...")
            self.IU_XRAY_ROOT = kagglehub.dataset_download("raddar/chest-xrays-indiana-university")
            print(f"IU-Xray downloaded to: {self.IU_XRAY_ROOT}")
            
            # MIMIC-CXR (Large, ~18GB)
            # CAUTION: This is huge. User warned about time.
            print("Downloading MIMIC-CXR (Warning: Large Download)...")
            self.MIMIC_CXR_ROOT = kagglehub.dataset_download("simhadrisadaram/mimic-cxr-dataset")
            print(f"MIMIC-CXR downloaded to: {self.MIMIC_CXR_ROOT}")
            
        except Exception as e:
            print(f"Error downloading datasets: {e}")
            print("Please ensure you have Kaggle API credentials set up if required, or download manually.")

config = Config()
