# LoRA Personalization Results

## Overview
This directory contains the results of personalizing a Pix2Struct model using LoRA (Low-Rank Adaptation) for a specific designer's sketching style.

## Contents

### Model Files
- `final_lora_adapter/` - The trained LoRA adapter weights (if training completed)
- `checkpoints/` - Training checkpoints
- `logs/` - Training logs and metrics

### Base Model
- The base model is loaded from: `models/checkpoints/last.ckpt`

### Evaluation Results
- `evaluation_results_*.json` - Metrics on different datasets
- `evaluation_report_*.html` - Interactive HTML reports

### Using the Trained Adapter

```python
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from peft import PeftModel

# Load base model and processor
base_model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-large")
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-large")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./outputs/final_lora_adapter")

# Generate code from sketch
encoding = processor(images=sketch_image, text="Generate website code", return_tensors="pt")
outputs = model.generate(**encoding, max_length=512)
generated_code = processor.decode(outputs[0], skip_special_tokens=True)
```

## Training Configuration
- **Base Model**: Loaded from checkpoint (models/checkpoints/last.ckpt)
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **Training Steps**: 2000
- **Designer Samples**: 10
- **Personalized Samples**: 50
- **Regularization Samples**: 200

Generated on: $(date)
