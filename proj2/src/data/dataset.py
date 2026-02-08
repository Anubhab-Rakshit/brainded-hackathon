import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms
from typing import List, Dict, Optional, Tuple

class RadiologyDataset(Dataset):
    """Base dataset class for Radiology Report Generation."""
    def __init__(self, tokenizer, correlation_matrix=None, transforms=None, max_seq_length=100):
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.max_seq_length = max_seq_length
        self.samples = [] # List of dicts: {'image_path': str, 'report': str, 'labels': List[int]}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image_path']
        report_text = sample['report']
        labels = torch.tensor(sample['labels'], dtype=torch.float32)

        # Load and Transform Image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            # Return a blank image if file not found (for mock/robustness)
            image = Image.new('RGB', (224, 224))
            
        if self.transforms:
            image = self.transforms(image)

        # Tokenize Report
        # Ensure tokenizer returns tensors
        tokenized_report = self.tokenizer(
            report_text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        
        input_ids = tokenized_report['input_ids'].squeeze(0)
        attention_mask = tokenized_report['attention_mask'].squeeze(0)

        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
class MockRadiologyDataset(RadiologyDataset):
    """Generates synthetic data for testing the pipeline without real datasets."""
    def __init__(self, num_samples=100, num_diseases=14, **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.num_diseases = num_diseases
        
        # Example dummy data
        for i in range(num_samples):
            self.samples.append({
                'image_path': f"mock_image_{i}.jpg", # Won't exist, base class handles it
                'report': "The lungs are clear. No pleural effusion or pneumothorax.",
                'labels': np.random.randint(0, 2, size=(num_diseases,)).tolist()
            })

class MIMICCXRDataset(RadiologyDataset):
    """Implementation for MIMIC-CXR Dataset."""
    def __init__(self, root_dir, csv_file, split='train', max_samples=None, **kwargs):
        super().__init__(**kwargs)
        self.root_dir = root_dir
        
        # Load CSV
        try:
            df = pd.read_csv(csv_file)
            # Filter by split if column exists
            if 'split' in df.columns:
                df = df[df['split'] == split]
            
            # Hackathon Optimization: Limit samples if configured
            if max_samples is not None:
                print(f"Dataset limited to {max_samples} samples for speed.")
                df = df.head(max_samples)
            
            # The downloaded CSV from KaggleHub (mimic_cxr_aug_train.csv) has 'image' column
            # format: "['files/p10/p10000032/s50414267/02aa804e....jpg']" (string representation of list)
            # We need to parse this.
            
            for _, row in df.iterrows():
                img_path = ""
                # Check for 'image' column (KaggleHub version)
                if 'image' in row:
                    raw_img_path = row['image']
                    # Clean the string list formatting "['path']" -> "path"
                    if isinstance(raw_img_path, str):
                        raw_img_path = raw_img_path.strip("[]'\"")
                    
                    # The path in CSV is typically "files/pXX/..."
                    # We need to prepend the root_dir and potentially 'official_data_iccv_final' if not included
                    # Based on inspection: CSV has "files/...", directory is "official_data_iccv_final/files/..."
                    # So we construct: root_dir / official_data_iccv_final / raw_img_path (which starts with files/)
                    
                    # Check if 'official_data_iccv_final' is needed
                    potential_path = os.path.join(root_dir, "official_data_iccv_final", raw_img_path)
                    if os.path.exists(potential_path):
                        img_path = potential_path
                    else:
                        # Fallback: maybe root_dir already points deeper?
                        img_path = os.path.join(root_dir, raw_img_path)
                
                # Fallback to old logic (subject_id/dicom_id) if 'image' not found
                elif 'subject_id' in row and 'dicom_id' in row:
                    img_path = os.path.join(root_dir, f"p{str(row['subject_id'])[:2]}", f"p{row['subject_id']}", f"s{row['study_id']}", f"{row['dicom_id']}.jpg")
                
                if img_path:
                    self.samples.append({
                        'image_path': img_path,
                        'report': row['text'] if pd.notna(row['text']) else "",
                        'labels': [0] * 14 # Placeholder
                    })
        except FileNotFoundError:
            print(f"Warning: CSV file {csv_file} not found. Dataset is empty.")

class IUXrayDataset(RadiologyDataset):
    """Implementation for IU-Xray Dataset (Benchmarking)."""
    def __init__(self, root_dir, csv_file, **kwargs):
        super().__init__(**kwargs)
        self.root_dir = root_dir
        # Logic to parse IU-Xray XMLs or pre-processed CSV
        # Similar to MIMIC-CXR
        pass

def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

# Simple Tokenizer Wrapper (using HuggingFace)
from transformers import AutoTokenizer

def get_tokenizer(model_name="bert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name)
