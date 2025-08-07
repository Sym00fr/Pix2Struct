#!/bin/bash

# LoRA Personalization Workflow Script
# This script automates the complete LoRA personalization pipeline

set -e  # Exit on any error

echo "🎨 Starting LoRA Personalization Pipeline"
echo "========================================"

# Configuration
DATA_DIR="data/original"
OUTPUT_DIR="data/designer_personalized_data"
OUTPUTS_DIR="outputs"

# Create necessary directories
echo "📁 Setting up directories..."
mkdir -p "$OUTPUTS_DIR"/{checkpoints,logs,final_lora_adapter}
mkdir -p "$DATA_DIR"

# Create a simple dataset metadata if it doesn't exist
if [ ! -f "$DATA_DIR/dataset_metadata.json" ]; then
    echo "📋 Creating simple dataset metadata..."
    cat > "$DATA_DIR/dataset_metadata.json" << EOF
{
  "train": [
    {"id": 0, "image_path": "train/images/sample_0.png", "sketch_path": "train/sketches/sample_0.png", "code_path": "train/code/sample_0.json", "split": "train"},
    {"id": 1, "image_path": "train/images/sample_1.png", "sketch_path": "train/sketches/sample_1.png", "code_path": "train/code/sample_1.json", "split": "train"},
    {"id": 2, "image_path": "train/images/sample_2.png", "sketch_path": "train/sketches/sample_2.png", "code_path": "train/code/sample_2.json", "split": "train"},
    {"id": 3, "image_path": "train/images/sample_3.png", "sketch_path": "train/sketches/sample_3.png", "code_path": "train/code/sample_3.json", "split": "train"},
    {"id": 4, "image_path": "train/images/sample_4.png", "sketch_path": "train/sketches/sample_4.png", "code_path": "train/code/sample_4.json", "split": "train"},
    {"id": 5, "image_path": "train/images/sample_5.png", "sketch_path": "train/sketches/sample_5.png", "code_path": "train/code/sample_5.json", "split": "train"},
    {"id": 6, "image_path": "train/images/sample_6.png", "sketch_path": "train/sketches/sample_6.png", "code_path": "train/code/sample_6.json", "split": "train"},
    {"id": 7, "image_path": "train/images/sample_7.png", "sketch_path": "train/sketches/sample_7.png", "code_path": "train/code/sample_7.json", "split": "train"},
    {"id": 8, "image_path": "train/images/sample_8.png", "sketch_path": "train/sketches/sample_8.png", "code_path": "train/code/sample_8.json", "split": "train"},
    {"id": 9, "image_path": "train/images/sample_9.png", "sketch_path": "train/sketches/sample_9.png", "code_path": "train/code/sample_9.json", "split": "train"}
  ],
  "val": [],
  "test": []
}
EOF
fi

# Step 1: Setup personalized dataset
echo ""
echo "🔧 Step 1: Setting up personalized dataset..."
python personalized_lora_setup.py \
    --data_dir "$DATA_DIR" \
    --designer_samples 5 \
    --augmented_samples 50 \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "✅ Dataset setup completed successfully!"
else
    echo "❌ Dataset setup failed! Continuing with simplified setup..."
    
    # Create minimal structure for demo
    mkdir -p "$OUTPUT_DIR"/{designer_original,personalized_train,regularization_data}/{images,sketches,code}
    mkdir -p "$OUTPUT_DIR/designer_webpages"
    
    # Create minimal metadata
    cat > "$OUTPUT_DIR/lora_dataset_metadata.json" << EOF
{
  "designer_samples": [],
  "personalized_train": [],
  "regularization": [],
  "config": {
    "designer_samples_count": 0,
    "personalized_samples_count": 0,
    "regularization_samples_count": 0,
    "total_samples": 0
  }
}
EOF
    
    # Create simple webpage
    cat > "$OUTPUT_DIR/designer_webpages/index.html" << EOF
<!DOCTYPE html>
<html>
<head><title>Designer Samples - Setup Required</title></head>
<body>
<h1>📋 LoRA Personalization Setup</h1>
<p>The personalized dataset setup needs to be completed manually.</p>
<p>Please ensure you have the original data with the correct structure.</p>
</body>
</html>
EOF
fi

# Step 2: Train LoRA adapter (only if we have data)
echo ""
echo "🚀 Step 2: Training LoRA adapter..."

if [ -f "$OUTPUT_DIR/lora_dataset_metadata.json" ]; then
    python train_personalized_lora.py
    
    if [ $? -eq 0 ]; then
        echo "✅ LoRA training completed successfully!"
    else
        echo "❌ LoRA training failed!"
        exit 1
    fi
else
    echo "⚠️  Skipping training - no dataset available"
fi
