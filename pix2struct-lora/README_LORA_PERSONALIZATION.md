# Personalized LoRA for Sketch-to-Code: Designer Style Adaptation

This project transforms a general Pix2Struct sketch-to-code model into a personalized system that adapts to a specific designer's sketching style using LoRA (Low-Rank Adaptation).

## 🎯 Overview

The personalization approach works by:

1. **Extracting Designer Samples**: Taking the first 10 sketches from your dataset as the target designer's style
2. **Component Extraction**: Using bounding box annotations to extract individual UI components from designer sketches
3. **Style Transfer**: Substituting designer components into other training samples to create personalized augmented data
4. **LoRA Training**: Training a lightweight LoRA adapter that personalizes the model while maintaining general capabilities through regularization

## 🏗️ Project Structure

```
pix2struct-lora/
├── personalized_lora_setup.py      # Dataset preparation for LoRA training
├── train_personalized_lora.py      # LoRA training script
├── evaluate_lora.py                # Evaluation script for trained models
├── configs/
│   └── lora_config.yaml            # LoRA training configuration
├── lora_personalized_data/         # Generated personalized dataset
│   ├── designer_original/          # Original 10 designer sketches
│   ├── designer_components/        # Extracted components library
│   ├── personalized_train/         # Augmented training data
│   ├── regularization_data/        # Original data for regularization
│   └── lora_configs/              # LoRA-specific configs
└── outputs/
    └── final_lora_adapter/         # Trained LoRA adapter
```

## 🚀 Quick Start

### Step 1: Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Personalized Dataset

```bash
# Extract designer samples and create augmented dataset
python personalized_lora_setup.py \
    --data_dir data \
    --designer_samples 10 \
    --augmented_samples 100 \
    --output_dir lora_personalized_data
```

This script will:
- Extract the first 10 samples as designer's style reference
- Extract individual components using bounding box annotations
- Create 100 augmented samples with designer style components
- Prepare regularization data from original samples

### Step 3: Train LoRA Adapter

```bash
# Train the personalized LoRA adapter
python train_personalized_lora.py
```

Or with custom configuration:
```bash
# Using Hydra configuration
python train_lora.py --config-path configs --config-name lora_config
```

### Step 4: Evaluate the Model

```bash
# Evaluate on a single sketch
python evaluate_lora.py \
    --lora_adapter outputs/final_lora_adapter \
    --single_sketch path/to/test_sketch.png

# Compare with base model
python evaluate_lora.py \
    --lora_adapter outputs/final_lora_adapter \
    --single_sketch path/to/test_sketch.png \
    --compare_base

# Full dataset evaluation
python evaluate_lora.py \
    --lora_adapter outputs/final_lora_adapter \
    --data_dir lora_personalized_data \
    --output_dir evaluation_results
```

## 🔧 Configuration

### LoRA Parameters

Key LoRA settings in `configs/lora_config.yaml`:

```yaml
lora:
  r: 16                    # LoRA rank (higher = more parameters)
  alpha: 32               # LoRA alpha (scaling factor)
  dropout: 0.1            # LoRA dropout
  target_modules:         # Which modules to adapt   !! Questi target module sono sbagliati !!
    - "q_proj"
    - "v_proj" 
    - "k_proj"
    - "out_proj"
    - "fc1"
    - "fc2"
    - "dense"
```

### Data Mixing Weights

Control how much each data type influences training:

```yaml
data:
  designer_weight: 3.0        # Higher weight for designer samples
  personalized_weight: 2.0    # Medium weight for personalized samples  
  regularization_weight: 1.0  # Normal weight for regularization samples
```

### Training Configuration

```yaml
training:
  batch_size: 4
  learning_rate: 5e-4
  max_steps: 2000
  gradient_accumulation_steps: 4
  early_stopping_patience: 5
```

## 📊 Dataset Structure

### Designer Original Samples
- **Location**: `lora_personalized_data/designer_original/`
- **Purpose**: Reference samples showing target designer's style
- **Count**: 10 samples (configurable)

### Designer Components Library
- **Location**: `lora_personalized_data/designer_components/`
- **Purpose**: Individual UI components extracted from designer sketches
- **Extraction**: Based on bounding box annotations in code data
- **Types**: buttons, labels, text fields, sliders, etc.

### Personalized Training Data
- **Location**: `lora_personalized_data/personalized_train/`
- **Purpose**: Augmented samples with designer style components
- **Generation**: Using `substitute_sketches` function from original codebase
- **Count**: 100 samples (configurable)

### Regularization Data
- **Location**: `lora_personalized_data/regularization_data/`
- **Purpose**: Prevent overfitting and maintain general capabilities
- **Source**: Random samples from original dataset

## 🎨 How Style Personalization Works

### 1. Component Extraction
```python
# Extract components based on bounding boxes
for annotation in code_data['annotations']:
    shape_attrs = annotation['shape_attributes']
    region_attrs = annotation['region_attributes']
    
    # Extract component region from sketch
    x, y, w, h = shape_attrs['x'], shape_attrs['y'], shape_attrs['width'], shape_attrs['height']
    component_img = sketch_img[y:y+h, x:x+w]
    
    # Save component with type information
    component_library.append({
        'type': region_attrs['type'],
        'component_path': comp_path,
        'dimensions': {'width': w, 'height': h}
    })
```

### 2. Style Transfer
```python
# Substitute designer components into other samples
personalized_sketch = substitute_sketches(
    original_image_path,
    regions,
    designer_components_df,
    components_map,
    only_rect_types,
    k=5  # Use fewer similar components for consistent style
)
```

### 3. Weighted Training
```python
# Different weights for different data types
weights = {
    'designer': 3.0,      # Highest weight - learn the style
    'personalized': 2.0,  # Medium weight - apply the style
    'regularization': 1.0 # Normal weight - maintain generalization
}
```

## 🔬 Evaluation Metrics

The evaluation script provides several metrics:

### Generation Quality
- **HTML Length**: Average length of generated HTML
- **CSS Length**: Average length of generated CSS
- **Syntax Validity**: Whether generated code is syntactically correct

### Style Consistency
- **Component Similarity**: How similar generated components are to designer style
- **Layout Patterns**: Whether the model captures designer's layout preferences

### Comparison with Base Model
- **Side-by-side Generation**: Compare LoRA vs base model outputs
- **Style Preservation**: How well the LoRA model maintains designer characteristics

## 🛠️ Advanced Usage

### Custom Component Types

Add new component types by modifying the components map:

```python
components_map = {
    "label": "label", 
    "text": "text_field", 
    "slider": "slider",
    "custom_widget": "custom_component",  # Add your custom type
    # ... more types
}
```

### Multi-Designer Training

Train on multiple designers by running the setup script multiple times:

```bash
# Designer A
python personalized_lora_setup.py --output_dir lora_designer_a --designer_samples 10

# Designer B  
python personalized_lora_setup.py --output_dir lora_designer_b --designer_samples 10
```

### Fine-tuning Hyperparameters

Adjust LoRA rank based on your needs:
- **r=8**: Faster training, less adaptation capability
- **r=16**: Good balance (recommended)
- **r=32**: More parameters, stronger adaptation

## 📈 Expected Results

After training, you should see:

1. **Designer Style Adoption**: Generated sketches should exhibit the designer's characteristic style
2. **Component Consistency**: UI elements should match the designer's drawing patterns
3. **Layout Preferences**: The model should prefer layouts similar to the designer's style
4. **Maintained Generalization**: The model should still work on general sketches

## 🐛 Troubleshooting

### Common Issues

**Memory Issues**:
```bash
# Reduce batch size and image resolution
# In configs/lora_config.yaml:
training:
  batch_size: 2  # Reduce from 4
data:
  image_size: [256, 256]  # Reduce from [384, 384]
```

**Poor Style Adaptation**:
```bash
# Increase designer weight and LoRA rank
lora:
  r: 32  # Increase from 16
data:
  designer_weight: 5.0  # Increase from 3.0
```

**Overfitting**:
```bash
# Increase regularization weight
data:
  regularization_weight: 2.0  # Increase from 1.0
```

## 📚 References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding](https://arxiv.org/abs/2210.03347)
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)

## 🤝 Contributing

Feel free to contribute by:
- Adding new component extraction methods
- Improving style transfer algorithms
- Adding evaluation metrics
- Creating better data augmentation techniques

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 