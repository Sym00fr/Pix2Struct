#!/usr/bin/env python3
"""
Evaluation script for personalized LoRA adapter
"""

import os
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List
import logging
import argparse
from tqdm import tqdm

from transformers import (
    Pix2StructForConditionalGeneration,
    Pix2StructProcessor
)
from peft import PeftModel
import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoRAEvaluator:
    """Evaluator for LoRA personalized models"""
    
    def __init__(self, base_model_name: str, lora_adapter_path: str):
        self.processor = Pix2StructProcessor.from_pretrained(base_model_name)
        
        # Load base model
        self.base_model = Pix2StructForConditionalGeneration.from_pretrained(base_model_name)
        
        # Load LoRA adapter
        self.lora_model = PeftModel.from_pretrained(self.base_model, lora_adapter_path)
        
        # Load metrics
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")
        
        logger.info("Models and metrics loaded successfully")
    
    def generate_code(self, sketch_image: Image.Image, model_type: str = "lora") -> str:
        """Generate code from sketch using specified model"""
        
        # Resize image
        original_width, original_height = sketch_image.size
        max_size = 384
        
        if max(original_width, original_height) > max_size:
            scale_factor = max_size / max(original_width, original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            sketch_image = sketch_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Process input
        encoding = self.processor(
            images=sketch_image,
            text="Generate website code from this sketch",
            return_tensors="pt",
            max_patches=96
        )
        
        # Generate
        model = self.lora_model if model_type == "lora" else self.base_model
        
        with torch.no_grad():
            generated_ids = model.generate(
                flattened_patches=encoding['flattened_patches'],
                attention_mask=encoding['attention_mask'],
                max_length=512,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
        
        # Decode
        generated_text = self.processor.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        
        return generated_text
    
    def evaluate_sample(self, sketch_path: str, ground_truth: str) -> Dict:
        """Evaluate a single sample"""
        
        sketch_image = Image.open(sketch_path).convert('RGB')
        
        # Generate with both models
        lora_output = self.generate_code(sketch_image, "lora")
        base_output = self.generate_code(sketch_image, "base")
        
        # Calculate metrics
        results = {
            'lora_output': lora_output,
            'base_output': base_output,
            'ground_truth': ground_truth
        }
        
        # BLEU scores
        lora_bleu = self.bleu_metric.compute(
            predictions=[lora_output],
            references=[[ground_truth]]
        )['bleu']
        
        base_bleu = self.bleu_metric.compute(
            predictions=[base_output],
            references=[[ground_truth]]
        )['bleu']
        
        # ROUGE scores
        lora_rouge = self.rouge_metric.compute(
            predictions=[lora_output],
            references=[ground_truth]
        )
        
        base_rouge = self.rouge_metric.compute(
            predictions=[base_output],
            references=[ground_truth]
        )
        
        # Exact match
        lora_exact = 1 if lora_output.strip() == ground_truth.strip() else 0
        base_exact = 1 if base_output.strip() == ground_truth.strip() else 0
        
        results.update({
            'lora_bleu': lora_bleu,
            'base_bleu': base_bleu,
            'lora_rouge1': lora_rouge['rouge1'],
            'base_rouge1': base_rouge['rouge1'],
            'lora_exact': lora_exact,
            'base_exact': base_exact
        })
        
        return results
    
    def evaluate_dataset(self, data_dir: str, dataset_type: str = "designer") -> Dict:
        """Evaluate on a dataset"""
        
        data_path = Path(data_dir)
        
        # Load metadata
        with open(data_path / 'lora_dataset_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        if dataset_type == "designer":
            samples = metadata['designer_samples']
        elif dataset_type == "personalized":
            samples = metadata['personalized_train'][:20]  # Limit for efficiency
        elif dataset_type == "regularization":
            samples = metadata['regularization'][:20]  # Limit for efficiency
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")
        
        results = []
        aggregate_metrics = {
            'lora_bleu': [], 'base_bleu': [],
            'lora_rouge1': [], 'base_rouge1': [],
            'lora_exact': [], 'base_exact': []
        }
        
        for sample in tqdm(samples, desc=f"Evaluating {dataset_type} samples"):
            try:
                # Load ground truth
                code_path = data_path / sample['code_path']
                with open(code_path, 'r') as f:
                    code_data = json.load(f)
                
                html_content = code_data['html']
                css_content = code_data['css']
                ground_truth = f"<html>{html_content}</html><css>{css_content}</css>"
                
                # Evaluate
                sketch_path = data_path / sample['sketch_path']
                result = self.evaluate_sample(str(sketch_path), ground_truth)
                result['sample_id'] = sample['id']
                results.append(result)
                
                # Aggregate metrics
                for metric in aggregate_metrics:
                    aggregate_metrics[metric].append(result[metric])
                
            except Exception as e:
                logger.warning(f"Error evaluating sample {sample['id']}: {e}")
                continue
        
        # Calculate averages
        averaged_metrics = {}
        for metric, values in aggregate_metrics.items():
            if values:
                averaged_metrics[f"avg_{metric}"] = np.mean(values)
                averaged_metrics[f"std_{metric}"] = np.std(values)
        
        return {
            'individual_results': results,
            'aggregate_metrics': averaged_metrics,
            'total_samples': len(results)
        }
    
    def create_evaluation_report(self, results: Dict, output_path: str):
        """Create HTML evaluation report"""
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LoRA Personalization Evaluation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #3498db;
        }}
        .metric-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }}
        .metric-comparison {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }}
        .lora-score {{
            color: #27ae60;
            font-weight: bold;
        }}
        .base-score {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .improvement {{
            background-color: #d5f4e6;
            color: #27ae60;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
        }}
        .samples-section {{
            margin-top: 30px;
        }}
        .sample-card {{
            background-color: #f8f9fa;
            margin-bottom: 20px;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #dee2e6;
        }}
        .sample-header {{
            background-color: #3498db;
            color: white;
            padding: 15px;
            font-weight: bold;
        }}
        .sample-content {{
            padding: 20px;
        }}
        .output-section {{
            margin-bottom: 20px;
        }}
        .output-title {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .output-text {{
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            font-family: monospace;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 LoRA Personalization Evaluation Report</h1>
            <p>Comparison of personalized LoRA adapter vs. base Pix2Struct model</p>
        </div>
        
        <div class="metrics-grid">
"""
        
        # Add metric cards
        metrics = results['aggregate_metrics']
        
        bleu_improvement = ((metrics.get('avg_lora_bleu', 0) - metrics.get('avg_base_bleu', 0)) / max(metrics.get('avg_base_bleu', 0.001), 0.001)) * 100
        rouge_improvement = ((metrics.get('avg_lora_rouge1', 0) - metrics.get('avg_base_rouge1', 0)) / max(metrics.get('avg_base_rouge1', 0.001), 0.001)) * 100
        exact_improvement = ((metrics.get('avg_lora_exact', 0) - metrics.get('avg_base_exact', 0)) / max(metrics.get('avg_base_exact', 0.001), 0.001)) * 100
        
        html_content += f"""
            <div class="metric-card">
                <div class="metric-title">BLEU Score</div>
                <div class="metric-comparison">
                    <span>LoRA: <span class="lora-score">{metrics.get('avg_lora_bleu', 0):.3f}</span></span>
                    <span>Base: <span class="base-score">{metrics.get('avg_base_bleu', 0):.3f}</span></span>
                </div>
                <div class="improvement">Improvement: {bleu_improvement:+.1f}%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">ROUGE-1 Score</div>
                <div class="metric-comparison">
                    <span>LoRA: <span class="lora-score">{metrics.get('avg_lora_rouge1', 0):.3f}</span></span>
                    <span>Base: <span class="base-score">{metrics.get('avg_base_rouge1', 0):.3f}</span></span>
                </div>
                <div class="improvement">Improvement: {rouge_improvement:+.1f}%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Exact Match</div>
                <div class="metric-comparison">
                    <span>LoRA: <span class="lora-score">{metrics.get('avg_lora_exact', 0):.3f}</span></span>
                    <span>Base: <span class="base-score">{metrics.get('avg_base_exact', 0):.3f}</span></span>
                </div>
                <div class="improvement">Improvement: {exact_improvement:+.1f}%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Total Samples</div>
                <div style="font-size: 24px; font-weight: bold; color: #3498db; text-align: center;">
                    {results['total_samples']}
                </div>
            </div>
        </div>
        
        <div class="samples-section">
            <h2>Sample Comparisons</h2>
"""
        
        # Add sample comparisons (limit to first 5 for brevity)
        for i, sample in enumerate(results['individual_results'][:5]):
            html_content += f"""
            <div class="sample-card">
                <div class="sample-header">
                    Sample {sample['sample_id']} - BLEU: LoRA {sample['lora_bleu']:.3f} vs Base {sample['base_bleu']:.3f}
                </div>
                <div class="sample-content">
                    <div class="output-section">
                        <div class="output-title">🎯 LoRA Output (Personalized)</div>
                        <div class="output-text">{sample['lora_output'][:500]}...</div>
                    </div>
                    
                    <div class="output-section">
                        <div class="output-title">🏠 Base Model Output</div>
                        <div class="output-text">{sample['base_output'][:500]}...</div>
                    </div>
                    
                    <div class="output-section">
                        <div class="output-title">📝 Ground Truth</div>
                        <div class="output-text">{sample['ground_truth'][:500]}...</div>
                    </div>
                </div>
            </div>
"""
        
        html_content += """
        </div>
    </div>
</body>
</html>
"""
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Evaluation report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA personalized model")
    parser.add_argument("--base_model", type=str, default="google/pix2struct-large", help="Base model name")
    parser.add_argument("--lora_adapter", type=str, default="outputs/final_lora_adapter", help="LoRA adapter path")
    parser.add_argument("--data_dir", type=str, default="data/designer_personalized_data", help="Data directory")
    parser.add_argument("--dataset_type", type=str, default="designer", choices=["designer", "personalized", "regularization"], help="Dataset to evaluate on")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = LoRAEvaluator(args.base_model, args.lora_adapter)
    
    # Run evaluation
    logger.info(f"Evaluating on {args.dataset_type} dataset...")
    results = evaluator.evaluate_dataset(args.data_dir, args.dataset_type)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    with open(output_dir / f"evaluation_results_{args.dataset_type}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create HTML report
    report_path = output_dir / f"evaluation_report_{args.dataset_type}.html"
    evaluator.create_evaluation_report(results, str(report_path))
    
    # Print summary
    metrics = results['aggregate_metrics']
    logger.info("📊 Evaluation Summary:")
    logger.info(f"   LoRA BLEU: {metrics.get('avg_lora_bleu', 0):.3f} (±{metrics.get('std_lora_bleu', 0):.3f})")
    logger.info(f"   Base BLEU: {metrics.get('avg_base_bleu', 0):.3f} (±{metrics.get('std_base_bleu', 0):.3f})")
    logger.info(f"   LoRA ROUGE-1: {metrics.get('avg_lora_rouge1', 0):.3f} (±{metrics.get('std_lora_rouge1', 0):.3f})")
    logger.info(f"   Base ROUGE-1: {metrics.get('avg_base_rouge1', 0):.3f} (±{metrics.get('std_base_rouge1', 0):.3f})")
    logger.info(f"   LoRA Exact Match: {metrics.get('avg_lora_exact', 0):.3f}")
    logger.info(f"   Base Exact Match: {metrics.get('avg_base_exact', 0):.3f}")
    logger.info(f"📁 Report saved to: {report_path}")

if __name__ == "__main__":
    main() 