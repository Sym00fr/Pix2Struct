#!/usr/bin/env python3
"""
Synthetic Designer Data Generator
This script generates synthetic variations of sketches using designer components.
The designer's style is naturally preserved through their original components.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import random
import logging
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DesignerComponentGenerator:
    """Generates synthetic data using designer components"""
    
    def __init__(self, designer_data_dir: str, train_data_dir: str, output_dir: str):
        self.designer_dir = Path(designer_data_dir)
        self.train_dir = Path(train_data_dir) 
        self.output_dir = Path(output_dir)
        
        # Load designer samples
        self.designer_components = self.extract_designer_components()
        logger.info(f"Extracted {len(self.designer_components)} designer components")
        
    def extract_designer_components(self) -> List[Dict]:
        """Extract components from designer sketches with their bounding boxes"""
        components = []
        
        for i in range(10):  # Designer samples 0-9
            code_file = self.designer_dir / "code" / f"designer_{i}.json"
            sketch_file = self.designer_dir / "sketches" / f"designer_{i}.png"
            
            if not code_file.exists() or not sketch_file.exists():
                continue
                
            try:
                with open(code_file, 'r') as f:
                    code_data = json.load(f)
                
                # Load sketch
                sketch = cv2.imread(str(sketch_file), cv2.IMREAD_GRAYSCALE)
                if sketch is None:
                    continue
                
                # Extract components based on annotations
                annotations = code_data.get('annotations', [])
                for j, annotation in enumerate(annotations):
                    if 'shape_attributes' not in annotation:
                        continue
                        
                    shape_attrs = annotation['shape_attributes']
                    region_attrs = annotation.get('region_attributes', {})
                    
                    x, y, w, h = shape_attrs.get('x', 0), shape_attrs.get('y', 0), \
                               shape_attrs.get('width', 0), shape_attrs.get('height', 0)
                    
                    if w <= 0 or h <= 0 or w < 10 or h < 10:
                        continue
                    
                    # Ensure bounds
                    x = max(0, min(x, sketch.shape[1]))
                    y = max(0, min(y, sketch.shape[0]))
                    w = min(w, sketch.shape[1] - x)
                    h = min(h, sketch.shape[0] - y)
                    
                    if w <= 10 or h <= 10:
                        continue
                    
                    # Extract component
                    component_img = sketch[y:y+h, x:x+w]
                    
                    components.append({
                        'type': region_attrs.get('type', 'unknown'),
                        'image': component_img,
                        'size': (w, h),
                        'designer_id': i,
                        'component_id': j,
                        'html': region_attrs.get('html', ''),
                        'css': region_attrs.get('css', '')
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing designer_{i}: {e}")
                continue
        
        return components
    

    
    def substitute_designer_components(self, base_sketch: np.ndarray, base_annotations: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """Substitute components in base sketch with designer components"""
        modified_sketch = base_sketch.copy()
        modified_annotations = []
        
        for annotation in base_annotations:
            if 'shape_attributes' not in annotation:
                modified_annotations.append(annotation)
                continue
                
            shape_attrs = annotation['shape_attributes']
            region_attrs = annotation.get('region_attributes', {})
            component_type = region_attrs.get('type', 'unknown')
            
            # Find matching designer component
            matching_components = [c for c in self.designer_components if c['type'] == component_type]
            
            if matching_components and random.random() < 0.6:  # 60% chance to substitute
                designer_comp = random.choice(matching_components)
                
                x, y, w, h = shape_attrs.get('x', 0), shape_attrs.get('y', 0), \
                           shape_attrs.get('width', 0), shape_attrs.get('height', 0)
                
                # Resize designer component to fit
                comp_img = designer_comp['image']
                if comp_img is not None and w > 10 and h > 10:
                    try:
                        # Resize component
                        resized_comp = cv2.resize(comp_img, (w, h))
                        
                        # Clear original area
                        modified_sketch[y:y+h, x:x+w] = 255
                        
                        # Place designer component
                        modified_sketch[y:y+h, x:x+w] = resized_comp
                        
                        # Update annotation with designer's HTML/CSS if available
                        new_annotation = annotation.copy()
                        if designer_comp['html']:
                            new_annotation['region_attributes']['html'] = designer_comp['html']
                        if designer_comp['css']:
                            new_annotation['region_attributes']['css'] = designer_comp['css']
                        
                        modified_annotations.append(new_annotation)
                    except Exception as e:
                        logger.warning(f"Error substituting component: {e}")
                        modified_annotations.append(annotation)
                else:
                    modified_annotations.append(annotation)
            else:
                modified_annotations.append(annotation)
        
        return modified_sketch, modified_annotations
    
    def generate_synthetic_sample(self, base_sample_id: str) -> Dict:
        """Generate a synthetic sample with designer components"""
        
        # Load base sample
        base_sketch_path = self.train_dir / "sketches" / f"{base_sample_id}.png"
        base_image_path = self.train_dir / "images" / f"{base_sample_id}.png"
        base_code_path = self.train_dir / "code" / f"{base_sample_id}.json"
        
        if not all([base_sketch_path.exists(), base_image_path.exists(), base_code_path.exists()]):
            return None
        
        try:
            # Load base data
            base_sketch = cv2.imread(str(base_sketch_path), cv2.IMREAD_GRAYSCALE)
            
            with open(base_code_path, 'r') as f:
                base_code_data = json.load(f)
            
            # Substitute components with designer components (no style transformation)
            final_sketch, modified_annotations = self.substitute_designer_components(
                base_sketch, base_code_data.get('annotations', [])
            )
            
            # Create modified code data
            modified_code_data = base_code_data.copy()
            modified_code_data['annotations'] = modified_annotations
            modified_code_data['source'] = 'designer_components_substituted'
            
            return {
                'sketch': final_sketch,
                'original_image': str(base_image_path),
                'code_data': modified_code_data,
                'base_sample_id': base_sample_id
            }
            
        except Exception as e:
            logger.warning(f"Error generating synthetic sample for {base_sample_id}: {e}")
            return None
    
    def generate_dataset(self, num_samples: int = 100) -> None:
        """Generate a synthetic dataset"""
        
        # Create output directories
        for subdir in ['sketches', 'images', 'code']:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Get list of available base samples
        base_sketches = list(self.train_dir.glob("sketches/sample_*.png"))
        base_sample_ids = [f.stem for f in base_sketches]
        
        # Remove designer samples from base samples
        designer_original_ids = ['sample_0', 'sample_1', 'sample_10', 'sample_100', 
                               'sample_101', 'sample_102', 'sample_103', 'sample_104', 
                               'sample_105', 'sample_106']
        base_sample_ids = [sid for sid in base_sample_ids if sid not in designer_original_ids]
        
        generated_count = 0
        metadata = []
        
        for i in tqdm(range(num_samples), desc="Generating synthetic samples"):
            if not base_sample_ids:
                break
                
            # Select random base sample
            base_sample_id = random.choice(base_sample_ids)
            
            # Generate synthetic sample
            synthetic_sample = self.generate_synthetic_sample(base_sample_id)
            
            if synthetic_sample is None:
                continue
            
            # Save synthetic sample
            synthetic_id = f"synthetic_{generated_count}"
            
            # Save sketch
            sketch_path = self.output_dir / "sketches" / f"{synthetic_id}.png"
            cv2.imwrite(str(sketch_path), synthetic_sample['sketch'])
            
            # Copy original image
            image_path = self.output_dir / "images" / f"{synthetic_id}.png"
            shutil.copy2(synthetic_sample['original_image'], image_path)
            
            # Save modified code
            code_path = self.output_dir / "code" / f"{synthetic_id}.json"
            with open(code_path, 'w') as f:
                json.dump(synthetic_sample['code_data'], f, indent=2)
            
            # Add to metadata
            metadata.append({
                'id': synthetic_id,
                'base_sample_id': synthetic_sample['base_sample_id'],
                'sketch_path': f"sketches/{synthetic_id}.png",
                'image_path': f"images/{synthetic_id}.png", 
                'code_path': f"code/{synthetic_id}.json",
                'type': 'designer_components_substituted'
            })
            
            generated_count += 1
        
        # Save metadata
        with open(self.output_dir / "synthetic_metadata.json", 'w') as f:
            json.dump({
                'synthetic_samples': metadata,
                'designer_style_source': str(self.designer_dir),
                'total_samples': generated_count,
                'generation_params': {
                    'num_designer_components': len(self.designer_components),
                    'substitution_probability': 0.6
                }
            }, f, indent=2)
        
        logger.info(f"✅ Generated {generated_count} synthetic samples with designer components")
        logger.info(f"📁 Saved to: {self.output_dir}")
        logger.info(f"📊 Metadata: {self.output_dir}/synthetic_metadata.json")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data with designer components")
    parser.add_argument("--designer_data", type=str, default="designer_sketches", 
                       help="Path to designer samples")
    parser.add_argument("--train_data", type=str, default="train", 
                       help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="synthetic_designer_data", 
                       help="Output directory")
    parser.add_argument("--num_samples", type=int, default=100, 
                       help="Number of synthetic samples to generate")
    
    args = parser.parse_args()
    
    # Create generator
    generator = DesignerComponentGenerator(
        args.designer_data, 
        args.train_data, 
        args.output_dir
    )
    
    # Generate synthetic dataset
    generator.generate_dataset(args.num_samples)
    
    logger.info("🎨 Synthetic designer component data generation complete!")

if __name__ == "__main__":
    main() 