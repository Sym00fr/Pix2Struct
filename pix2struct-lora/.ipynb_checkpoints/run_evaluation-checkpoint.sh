# Configuration
DATA_DIR="data/original"
OUTPUT_DIR="data/designer_personalized_data"
OUTPUTS_DIR="outputs"

# Step 3: Evaluate LoRA adapter (only if we have models)
echo ""
echo "📊 Step 3: Evaluating LoRA adapter..."


if [ -d "$OUTPUTS_DIR/final_lora_adapter" ]; then
    # Evaluate on designer samples
    echo "   Evaluating on designer samples..."
    python evaluate_lora.py \
        --dataset_type designer \
        --data_dir "$OUTPUT_DIR" \
        --output_dir "$OUTPUTS_DIR" || echo "Designer evaluation failed"
    
    # Evaluate on personalized samples  
    echo "   Evaluating on personalized samples..."
    python evaluate_lora.py \
        --dataset_type personalized \
        --data_dir "$OUTPUT_DIR" \
        --output_dir "$OUTPUTS_DIR" || echo "Personalized evaluation failed"
    
    echo "✅ Evaluation completed!"
else
    echo "⚠️  Skipping evaluation - no trained model available"
fi