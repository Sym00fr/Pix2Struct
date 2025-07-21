#!/usr/bin/env python3
"""
Simplified LoRA Training for Personalized Sketch-to-Code
"""

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List
import logging
import random

from transformers import (
    Pix2StructForConditionalGeneration,
    Pix2StructProcessor,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from torch.optim import AdamW
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonalizedSketchDataset(Dataset):
    """Dataset for personalized sketch-to-code training"""
    
    def __init__(self, data_dir: str, metadata_file: str, processor, data_type: str, max_length: int = 512):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.data_type = data_type
        self.max_length = max_length
        
        # Load metadata
        with open(self.data_dir / metadata_file, 'r') as f:
            metadata = json.load(f)
        
        if data_type == "designer":
            self.samples = metadata['designer_samples']
        elif data_type == "personalized":
            self.samples = metadata['personalized_train']
        elif data_type == "regularization":
            self.samples = metadata['regularization']
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
        
        logger.info(f"Loaded {len(self.samples)} samples for {data_type} dataset")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load sketch image
        sketch_path = self.data_dir / sample['sketch_path']
        sketch_image = Image.open(sketch_path).convert('RGB')
        
        # Resize image
        original_width, original_height = sketch_image.size
        max_size = 384
        
        if max(original_width, original_height) > max_size:
            scale_factor = max_size / max(original_width, original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            sketch_image = sketch_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Load code data
        code_path = self.data_dir / sample['code_path']
        with open(code_path, 'r', encoding='utf-8') as f:
            code_data = json.load(f)
        
        # Format target text
        html_content = code_data['html']
        css_content = code_data['css']
        
        # Truncate for efficiency
        if len(html_content) > 1500:
            html_content = html_content[:1500] + "..."
        if len(css_content) > 500:
            css_content = css_content[:500] + "..."
        
        target_text = f"<html>{html_content}</html><css>{css_content}</css>"
        
        # Process inputs
        encoding = self.processor(
            images=sketch_image,
            text="Generate website code from this sketch",
            return_tensors="pt",
            max_patches=96
        )
        
        # Process target
        target_encoding = self.processor.tokenizer(
            target_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'flattened_patches': encoding['flattened_patches'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze(),
            'data_type': self.data_type
        }

class WeightedConcatDataset(ConcatDataset):
    """Dataset that weights different data types"""
    
    def __init__(self, datasets, weights):
        super().__init__(datasets)
        self.weights = weights
        
        # Create weighted indices
        self.weighted_indices = []
        for dataset_idx, (dataset, weight) in enumerate(zip(datasets, weights)):
            dataset_indices = [(dataset_idx, i) for i in range(len(dataset))]
            repeated_indices = dataset_indices * max(1, int(weight))
            self.weighted_indices.extend(repeated_indices)
        
        random.shuffle(self.weighted_indices)
    
    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.weighted_indices[idx]
        return self.datasets[dataset_idx][sample_idx]
    
    def __len__(self):
        return len(self.weighted_indices)

class Pix2StructLoRATrainer(pl.LightningModule):
    """LoRA trainer for personalized Pix2Struct"""
    
    def __init__(self, model_name: str = "google/pix2struct-large", checkpoint_path: str = None, learning_rate: float = 5e-4, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        # Initialize processor
        self.processor = Pix2StructProcessor.from_pretrained(model_name)
        
        # Initialize base model (from checkpoint if available)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading model from checkpoint: {checkpoint_path}")
            # Load from checkpoint
            import pytorch_lightning as pl
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.base_model = Pix2StructForConditionalGeneration.from_pretrained(model_name)
            # Load the state dict from checkpoint if available
            if 'state_dict' in checkpoint:
                model_state_dict = {}
                for key, value in checkpoint['state_dict'].items():
                    if key.startswith('model.'):
                        model_state_dict[key[6:]] = value  # Remove 'model.' prefix
                if model_state_dict:
                    self.base_model.load_state_dict(model_state_dict, strict=False)
                    print("Loaded model weights from checkpoint")
        else:
            self.base_model = Pix2StructForConditionalGeneration.from_pretrained(model_name)
        
        # Prepare model for LoRA
        self.base_model = prepare_model_for_kbit_training(self.base_model)
        
        # Setup LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            #target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
            #prova con i moduli del pix2struct
            target_modules=["query", "value", "key", "output", "wi_0", "wi_1"],
            bias="none"
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.base_model, peft_config)
        
        logger.info(f"LoRA model created with {self.model.num_parameters()} total parameters")
        logger.info(f"Trainable parameters: {self.model.num_parameters(only_trainable=True)}")
        
    def forward(self, flattened_patches, attention_mask, labels=None):
        return self.model(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self(
            flattened_patches=batch['flattened_patches'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        # Weighted loss based on data type
        data_types = batch['data_type']
        weights = torch.tensor([
            3.0 if dt == 'designer' else 2.0 if dt == 'personalized' else 1.0 
            for dt in data_types
        ], device=self.device)
        
        weighted_loss = outputs.loss * weights.mean()
        
        self.log('train/loss', weighted_loss, prog_bar=True)
        return weighted_loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(
            flattened_patches=batch['flattened_patches'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        self.log('val/loss', outputs.loss, prog_bar=True, sync_dist=True)
        return outputs.loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=100, num_training_steps=2000
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

class PersonalizedLoRADataModule(pl.LightningDataModule):
    """Data module for LoRA training"""
    
    def __init__(self, data_dir: str, model_name: str, batch_size: int = 4, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.model_name = model_name
        self.batch_size = batch_size
        
        self.processor = Pix2StructProcessor.from_pretrained(model_name)
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Create datasets
            designer_dataset = PersonalizedSketchDataset(
                self.data_dir, "lora_dataset_metadata.json", self.processor, "designer"
            )
            personalized_dataset = PersonalizedSketchDataset(
                self.data_dir, "lora_dataset_metadata.json", self.processor, "personalized"
            )
            regularization_dataset = PersonalizedSketchDataset(
                self.data_dir, "lora_dataset_metadata.json", self.processor, "regularization"
            )
            
            # Create weighted dataset
            self.train_dataset = WeightedConcatDataset(
                [designer_dataset, personalized_dataset, regularization_dataset],
                [3.0, 2.0, 1.0]  # weights
            )
            
            self.val_dataset = regularization_dataset
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        collated = {}
        for key in batch[0].keys():
            if key == 'data_type':
                collated[key] = [item[key] for item in batch]
            else:
                collated[key] = torch.stack([item[key] for item in batch])
        return collated

def main():
    """Main training function"""
    
    # Set seed
    pl.seed_everything(42)
    
    # Data module
    data_module = PersonalizedLoRADataModule(
        data_dir="data/designer_personalized_data",
        model_name="google/pix2struct-large",
        batch_size=4
    )
    
    # Model
    model = Pix2StructLoRATrainer(
        model_name="google/pix2struct-large",
        checkpoint_path="models/checkpoints/last.ckpt",
        learning_rate=5e-4
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="outputs/checkpoints",
            filename="lora-{epoch:02d}-{val/loss:.2f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(monitor="val/loss", patience=5, mode="min")
    ]
    
    # Logger
    logger_instance = TensorBoardLogger(save_dir="outputs/logs", name="lora_personalized")
    
    # Trainer
    trainer = pl.Trainer(
        max_steps=2000,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        logger=logger_instance,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=0.25,
        accelerator="auto"
    )
    
    # Train
    logger.info("🚀 Starting LoRA personalized training...")
    trainer.fit(model, data_module)
    
    # Save final adapter
    final_path = Path("outputs/final_lora_adapter")
    final_path.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(final_path)
    model.processor.save_pretrained(final_path)
    
    logger.info(f"✅ Training complete! LoRA adapter saved to: {final_path}")

if __name__ == "__main__":
    main() 