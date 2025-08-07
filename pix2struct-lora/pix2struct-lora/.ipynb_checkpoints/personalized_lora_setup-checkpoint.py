#!/usr/bin/env python3
"""
Personalized LoRA Dataset Setup for Designer Style Adaptation
This script sets up the data structure for training a LoRA adapter on a specific designer's sketching style
"""

import os
import json
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import random
import logging
from tqdm import tqdm
import argparse
import sys

# Add parent directory to path to import generate_dataset functions
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from generate_dataset import substitute_sketches, DatasetConfig
except ImportError:
    print("Warning: Could not import from generate_dataset. Some functions may not be available.")
    substitute_sketches = None
    DatasetConfig = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonalizedLoRADatasetManager:
    """Manager for creating personalized LoRA datasets"""
    
    def __init__(self, 
                 data_dir: str, 
                 designer_samples: int = 10,
                 augmented_samples: int = 100,
                 output_dir: str = "designer_personalized_data"):
        self.data_dir = Path(data_dir)
        self.designer_samples = designer_samples
        self.augmented_samples = augmented_samples
        self.output_dir = Path(output_dir)
        
        # Load metadata
        with open(self.data_dir / 'dataset_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.train_samples = self.metadata['train']
        
    def setup_directories(self):
        """Create necessary directories for LoRA training"""
        directories = [
            'designer_original',      # Original 10 designer sketches
            'designer_components',    # Extracted components from designer sketches
            'personalized_train',     # Augmented training data with designer style
            'regularization_data',    # Original data for regularization
            'designer_webpages'       # HTML webpages showing designer samples
        ]
        
        for dir_name in directories:
            for subdir in ['images', 'sketches', 'code']:
                (self.output_dir / dir_name / subdir).mkdir(parents=True, exist_ok=True)
        
        # Create webpage directory without subdirs
        (self.output_dir / 'designer_webpages').mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created directory structure in {self.output_dir}")
    
    def extract_designer_samples(self):
        """Extract the first 10 samples as designer's personal style"""
        designer_metadata = []
        
        # Take first 10 samples (based on the order in metadata)
        first_10_samples = self.train_samples[:self.designer_samples]
        
        for i, sample in enumerate(first_10_samples):
            # Copy original files
            src_sketch = self.data_dir / sample['sketch_path']
            src_image = self.data_dir / sample['image_path'] 
            src_code = self.data_dir / sample['code_path']
            
            dst_sketch = self.output_dir / 'designer_original' / 'sketches' / f'designer_{i}.png'
            dst_image = self.output_dir / 'designer_original' / 'images' / f'designer_{i}.png'
            dst_code = self.output_dir / 'designer_original' / 'code' / f'designer_{i}.json'
            
            # Check if source files exist
            if not src_sketch.exists():
                logger.warning(f"Sketch not found: {src_sketch}")
                continue
            if not src_image.exists():
                logger.warning(f"Image not found: {src_image}")
                continue
            if not src_code.exists():
                logger.warning(f"Code not found: {src_code}")
                continue
            
            shutil.copy2(src_sketch, dst_sketch)
            shutil.copy2(src_image, dst_image)
            shutil.copy2(src_code, dst_code)
            
            designer_metadata.append({
                'id': f'designer_{i}',
                'original_id': sample['id'],
                'sketch_path': str(dst_sketch.relative_to(self.output_dir)),
                'image_path': str(dst_image.relative_to(self.output_dir)),
                'code_path': str(dst_code.relative_to(self.output_dir))
            })
        
        # Save designer metadata
        with open(self.output_dir / 'designer_metadata.json', 'w') as f:
            json.dump(designer_metadata, f, indent=2)
        
        logger.info(f"Extracted {len(designer_metadata)} designer samples")
        return designer_metadata
    
    def create_designer_webpages(self, designer_metadata: List[Dict]):
        """Create HTML webpages showing the designer's sketches and code"""
        
        # Create individual pages for each sample
        for i, sample in enumerate(designer_metadata):
            # Load the code data
            code_path = self.output_dir / sample['code_path']
            with open(code_path, 'r') as f:
                code_data = json.load(f)
            
            # Create HTML page
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Designer Sample {i} - Original ID {sample['original_id']}</title>
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
        .images-section {{
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .image-container {{
            flex: 1;
            min-width: 300px;
        }}
        .image-container h3 {{
            color: #2c3e50;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 10px;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border: 2px solid #ecf0f1;
            border-radius: 5px;
        }}
        .code-section {{
            margin-top: 30px;
        }}
        .code-container {{
            margin-bottom: 20px;
        }}
        .code-container h3 {{
            color: #27ae60;
            background-color: #ecf0f1;
            padding: 10px;
            margin: 0;
            border-radius: 5px 5px 0 0;
        }}
        .code-container pre {{
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 0 0 5px 5px;
            overflow-x: auto;
            margin: 0;
        }}
        .navigation {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #bdc3c7;
        }}
        .nav-link {{
            display: inline-block;
            margin: 0 10px;
            padding: 10px 20px;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            transition: background-color 0.3s;
        }}
        .nav-link:hover {{
            background-color: #2980b9;
        }}
        .current {{
            background-color: #e74c3c;
        }}
        .stats {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .stats h4 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .stats p {{
            margin: 5px 0;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 Designer Sample {i}</h1>
            <p><strong>Original Sample ID:</strong> {sample['original_id']}</p>
            <p>This is one of the 10 reference samples representing the target designer's sketching style</p>
        </div>
        
        <div class="stats">
            <h4>📊 Sample Statistics</h4>
            <p><strong>HTML Length:</strong> {len(code_data.get('html', ''))} characters</p>
            <p><strong>CSS Length:</strong> {len(code_data.get('css', ''))} characters</p>
            <p><strong>Components:</strong> {len(code_data.get('annotations', []))} annotated elements</p>
        </div>
        
        <div class="images-section">
            <div class="image-container">
                <h3>🖼️ Original Website Screenshot</h3>
                <img src="../{sample['image_path']}" alt="Original Website Screenshot">
            </div>
            <div class="image-container">
                <h3>✏️ Designer Sketch</h3>
                <img src="../{sample['sketch_path']}" alt="Designer Sketch">
            </div>
        </div>
        
        <div class="code-section">
            <div class="code-container">
                <h3>📝 HTML Code</h3>
                <pre><code>{code_data.get('html', 'No HTML found')}</code></pre>
            </div>
            
            <div class="code-container">
                <h3>🎨 CSS Code</h3>
                <pre><code>{code_data.get('css', 'No CSS found')}</code></pre>
            </div>
        </div>
        
        <div class="navigation">
            <h4>Navigate Designer Samples</h4>
"""
            
            # Add navigation links
            for j in range(len(designer_metadata)):
                if j == i:
                    html_content += f'<a href="designer_sample_{j}.html" class="nav-link current">Sample {j}</a>\n'
                else:
                    html_content += f'<a href="designer_sample_{j}.html" class="nav-link">Sample {j}</a>\n'
            
            html_content += f'''
            <br><br>
            <a href="index.html" class="nav-link">📋 Back to Overview</a>
        </div>
    </div>
</body>
</html>
'''
            
            # Save individual page
            page_path = self.output_dir / 'designer_webpages' / f'designer_sample_{i}.html'
            with open(page_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        # Create index page
        self.create_designer_index_page(designer_metadata)
        
        logger.info(f"Created designer webpages in {self.output_dir}/designer_webpages/")
    
    def create_designer_index_page(self, designer_metadata: List[Dict]):
        """Create an index page showing all designer samples"""
        
        index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Designer Style Reference - LoRA Personalization</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .sample-card {
            border: 2px solid #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            transition: transform 0.3s, box-shadow 0.3s;
            background-color: white;
        }
        .sample-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .card-header {
            background-color: #3498db;
            color: white;
            padding: 15px;
            text-align: center;
        }
        .card-images {
            display: flex;
            height: 200px;
        }
        .card-image {
            flex: 1;
            background-size: cover;
            background-position: center;
            position: relative;
        }
        .image-label {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: rgba(0,0,0,0.7);
            color: white;
            padding: 5px;
            text-align: center;
            font-size: 12px;
        }
        .card-footer {
            padding: 15px;
            text-align: center;
        }
        .view-btn {
            background-color: #27ae60;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            transition: background-color 0.3s;
        }
        .view-btn:hover {
            background-color: #219a52;
        }
        .stats-section {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .stat-card {
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            border-left: 4px solid #3498db;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .stat-label {
            color: #7f8c8d;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 Designer Style Reference</h1>
            <p><strong>LoRA Personalization Project</strong></p>
            <p>These 10 samples represent the target designer's sketching style that will be used to personalize the Pix2Struct model using LoRA adaptation.</p>
        </div>
        
        <div class="stats-section">
            <h3>📊 Dataset Overview</h3>
            <div class="stats-grid">
"""
        
        # Calculate statistics
        total_samples = len(designer_metadata)
        total_html_chars = 0
        total_css_chars = 0
        total_components = 0
        
        for sample in designer_metadata:
            code_path = self.output_dir / sample['code_path']
            try:
                with open(code_path, 'r') as f:
                    code_data = json.load(f)
                total_html_chars += len(code_data.get('html', ''))
                total_css_chars += len(code_data.get('css', ''))
                total_components += len(code_data.get('annotations', []))
            except:
                pass
        
        index_html += f"""
                <div class="stat-card">
                    <div class="stat-value">{total_samples}</div>
                    <div class="stat-label">Designer Samples</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_components}</div>
                    <div class="stat-label">Total Components</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_html_chars:,}</div>
                    <div class="stat-label">HTML Characters</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_css_chars:,}</div>
                    <div class="stat-label">CSS Characters</div>
                </div>
            </div>
        </div>
        
        <div class="gallery">
"""
        
        # Add sample cards
        for i, sample in enumerate(designer_metadata):
            index_html += f"""
            <div class="sample-card">
                <div class="card-header">
                    <h3>Sample {i}</h3>
                    <p>Original ID: {sample['original_id']}</p>
                </div>
                <div class="card-images">
                    <div class="card-image" style="background-image: url('../{sample['image_path']}');">
                        <div class="image-label">Website Screenshot</div>
                    </div>
                    <div class="card-image" style="background-image: url('../{sample['sketch_path']}');">
                        <div class="image-label">Designer Sketch</div>
                    </div>
                </div>
                <div class="card-footer">
                    <a href="designer_sample_{i}.html" class="view-btn">View Details</a>
                </div>
            </div>
"""
        
        index_html += """
        </div>
        
        <div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7;">
            <h4>🚀 Next Steps</h4>
            <p>These samples will be used to:</p>
            <ol style="text-align: left; display: inline-block;">
                <li>Extract individual UI components based on bounding box annotations</li>
                <li>Create a personalized component library representing the designer's style</li>
                <li>Generate augmented training data by substituting designer components into other samples</li>
                <li>Train a LoRA adapter that personalizes the model to this specific sketching style</li>
            </ol>
        </div>
    </div>
</body>
</html>
"""
        
        # Save index page
        index_path = self.output_dir / 'designer_webpages' / 'index.html'
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_html)
    
    def extract_designer_components(self, designer_metadata: List[Dict]):
        """Extract individual components from designer sketches using bounding boxes"""
        component_library = []
        
        for designer_data in tqdm(designer_metadata, desc="Extracting components"):
            # Load code data to get bounding boxes
            code_path = self.output_dir / designer_data['code_path']
            with open(code_path, 'r') as f:
                code_data = json.load(f)
            
            # Load sketch image
            sketch_path = self.output_dir / designer_data['sketch_path']
            sketch_img = cv2.imread(str(sketch_path), cv2.IMREAD_GRAYSCALE)
            
            if sketch_img is None:
                logger.warning(f"Could not load sketch: {sketch_path}")
                continue
            
            # Extract components based on annotations
            annotations = code_data.get('annotations', [])
            
            for j, annotation in enumerate(annotations):
                shape_attrs = annotation['shape_attributes']
                region_attrs = annotation['region_attributes']
                
                # Skip invalid bounding boxes
                if shape_attrs['width'] <= 0 or shape_attrs['height'] <= 0:
                    continue
                
                # Extract component region
                x, y, w, h = (shape_attrs['x'], shape_attrs['y'], 
                            shape_attrs['width'], shape_attrs['height'])
                
                # Ensure bounds are within image
                x = max(0, min(x, sketch_img.shape[1]))
                y = max(0, min(y, sketch_img.shape[0]))
                w = min(w, sketch_img.shape[1] - x)
                h = min(h, sketch_img.shape[0] - y)
                
                if w <= 10 or h <= 10:  # Skip too small components
                    continue
                
                # Extract component
                component_img = sketch_img[y:y+h, x:x+w]
                
                # Save component
                comp_filename = f"{designer_data['id']}_comp_{j}_{region_attrs['type']}.png"
                comp_path = self.output_dir / 'designer_components' / 'sketches' / comp_filename
                cv2.imwrite(str(comp_path), component_img)
                
                # Store component metadata
                component_library.append({
                    'component_id': f"{designer_data['id']}_comp_{j}",
                    'type': region_attrs['type'],
                    'source_sketch': designer_data['id'],
                    'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                    'component_path': str(comp_path.relative_to(self.output_dir)),
                    'dimensions': {'width': w, 'height': h}
                })
        
        # Save component library
        with open(self.output_dir / 'designer_components_library.json', 'w') as f:
            json.dump(component_library, f, indent=2)
        
        logger.info(f"Extracted {len(component_library)} designer components")
        return component_library
    
    def create_augmented_dataset(self, component_library: List[Dict]):
        """Create augmented dataset by substituting designer components into other samples"""
        if substitute_sketches is None:
            logger.warning("substitute_sketches function not available. Skipping augmented dataset creation.")
            return []
        
        augmented_metadata = []
        
        # Load sketch data for substitution
        if DatasetConfig is None:
            logger.warning("DatasetConfig not available. Using default settings.")
            components_map = {
                "label": "label", "text": "text_field", "slider": "slider", "number": "text_field",
                "image": "image", "table": "data_table", "radio_checked": "radio_button_checked",
                "radio_unchecked": "radio_button_unchecked", "button": "button",
                "checkbox_switch": "switch_enabled", "checkbox_label": "label",
                "text_area": "text_area", "carousel": "carousel"
            }
            only_rect = ['form', 'footer', 'navbar', 'header', 'table', 'sidebar', 'call_to_action', 'card']
        else:
            config = DatasetConfig()
            components_map = getattr(config, 'components_map', {})
            only_rect = getattr(config, 'only_rect', [])
        
        # Create a component mapping for substitution
        components_df_data = []
        for comp in component_library:
            comp_type = comp['type']
            components_df_data.append([
                comp_type, 
                str(self.output_dir / comp['component_path']), 
                (comp['dimensions']['width'], comp['dimensions']['height']), 
                'train'
            ])
        
        current_sketches_df = pd.DataFrame(components_df_data, columns=['category', 'sketch_path', 'dims', 'set'])
        
        # Select base samples for augmentation (excluding designer samples)
        base_samples = self.train_samples[self.designer_samples:self.designer_samples + self.augmented_samples]
        
        for i, sample in enumerate(tqdm(base_samples, desc="Creating augmented samples")):
            try:
                # Load original sample
                orig_sketch_path = self.data_dir / sample['sketch_path']
                orig_image_path = self.data_dir / sample['image_path']
                orig_code_path = self.data_dir / sample['code_path']
                
                if not all([orig_sketch_path.exists(), orig_image_path.exists(), orig_code_path.exists()]):
                    logger.warning(f"Missing files for sample {sample['id']}")
                    continue
                
                with open(orig_code_path, 'r') as f:
                    code_data = json.load(f)
                
                # Apply designer style substitution
                regions = code_data.get('annotations', [])
                
                # Use substitute_sketches to create personalized version
                personalized_sketch = substitute_sketches(
                    str(orig_image_path),
                    regions,
                    current_sketches_df,
                    components_map,
                    only_rect,
                    k=5  # Use fewer similar components for more consistent style
                )
                
                if personalized_sketch is not None:
                    # Save personalized sketch
                    pers_sketch_path = self.output_dir / 'personalized_train' / 'sketches' / f'personalized_{i}.png'
                    cv2.imwrite(str(pers_sketch_path), personalized_sketch)
                    
                    # Copy original image and code
                    pers_image_path = self.output_dir / 'personalized_train' / 'images' / f'personalized_{i}.png'
                    pers_code_path = self.output_dir / 'personalized_train' / 'code' / f'personalized_{i}.json'
                    
                    shutil.copy2(orig_image_path, pers_image_path)
                    shutil.copy2(orig_code_path, pers_code_path)
                    
                    augmented_metadata.append({
                        'id': f'personalized_{i}',
                        'original_id': sample['id'],
                        'sketch_path': str(pers_sketch_path.relative_to(self.output_dir)),
                        'image_path': str(pers_image_path.relative_to(self.output_dir)),
                        'code_path': str(pers_code_path.relative_to(self.output_dir)),
                        'style': 'designer_personalized'
                    })
                
            except Exception as e:
                logger.warning(f"Failed to create augmented sample {i}: {e}")
                continue
        
        # Save augmented metadata
        with open(self.output_dir / 'personalized_train_metadata.json', 'w') as f:
            json.dump(augmented_metadata, f, indent=2)
        
        logger.info(f"Created {len(augmented_metadata)} augmented personalized samples")
        return augmented_metadata
    
    def create_regularization_data(self):
        """Create regularization dataset from original data"""
        reg_metadata = []
        available_samples = self.train_samples[self.designer_samples + self.augmented_samples:]
        reg_samples = random.sample(available_samples, min(200, len(available_samples)))
        
        for i, sample in enumerate(tqdm(reg_samples, desc="Creating regularization data")):
            try:
                # Copy files for regularization
                src_sketch = self.data_dir / sample['sketch_path']
                src_image = self.data_dir / sample['image_path']
                src_code = self.data_dir / sample['code_path']
                
                if not all([src_sketch.exists(), src_image.exists(), src_code.exists()]):
                    logger.warning(f"Missing files for regularization sample {sample['id']}")
                    continue
                
                dst_sketch = self.output_dir / 'regularization_data' / 'sketches' / f'reg_{i}.png'
                dst_image = self.output_dir / 'regularization_data' / 'images' / f'reg_{i}.png'
                dst_code = self.output_dir / 'regularization_data' / 'code' / f'reg_{i}.json'
                
                shutil.copy2(src_sketch, dst_sketch)
                shutil.copy2(src_image, dst_image)
                shutil.copy2(src_code, dst_code)
                
                reg_metadata.append({
                    'id': f'reg_{i}',
                    'original_id': sample['id'],
                    'sketch_path': str(dst_sketch.relative_to(self.output_dir)),
                    'image_path': str(dst_image.relative_to(self.output_dir)),
                    'code_path': str(dst_code.relative_to(self.output_dir)),
                    'style': 'original'
                })
            except Exception as e:
                logger.warning(f"Failed to create regularization sample {i}: {e}")
                continue
        
        # Save regularization metadata
        with open(self.output_dir / 'regularization_metadata.json', 'w') as f:
            json.dump(reg_metadata, f, indent=2)
        
        logger.info(f"Created {len(reg_metadata)} regularization samples")
        return reg_metadata
    
    def create_combined_metadata(self, designer_metadata, augmented_metadata, reg_metadata):
        """Create combined metadata for LoRA training"""
        combined_metadata = {
            'designer_samples': designer_metadata,
            'personalized_train': augmented_metadata,
            'regularization': reg_metadata,
            'config': {
                'designer_samples_count': len(designer_metadata),
                'personalized_samples_count': len(augmented_metadata),
                'regularization_samples_count': len(reg_metadata),
                'total_samples': len(designer_metadata) + len(augmented_metadata) + len(reg_metadata)
            }
        }
        
        with open(self.output_dir / 'lora_dataset_metadata.json', 'w') as f:
            json.dump(combined_metadata, f, indent=2)
        
        logger.info("Created combined LoRA dataset metadata")
        return combined_metadata

def main():
    parser = argparse.ArgumentParser(description="Setup Personalized LoRA Dataset")
    parser.add_argument("--data_dir", type=str, default="data/original", help="Path to original dataset")
    parser.add_argument("--designer_samples", type=int, default=5, help="Number of designer samples")
    parser.add_argument("--augmented_samples", type=int, default=100, help="Number of augmented samples")
    parser.add_argument("--output_dir", type=str, default="data/designer_personalized_data", help="Output directory")
    
    args = parser.parse_args()
    
    # Create dataset manager
    manager = PersonalizedLoRADatasetManager(
        data_dir=args.data_dir,
        designer_samples=args.designer_samples,
        augmented_samples=args.augmented_samples,
        output_dir=args.output_dir
    )
    
    # Setup directories
    manager.setup_directories()
    
    # Extract designer samples
    designer_metadata = manager.extract_designer_samples()
    
    # Create designer webpages
    manager.create_designer_webpages(designer_metadata)
    
    # Extract designer components
    component_library = manager.extract_designer_components(designer_metadata)
    
    # Create augmented dataset
    augmented_metadata = manager.create_augmented_dataset(component_library)
    
    # Create regularization data
    reg_metadata = manager.create_regularization_data()
    
    # Create combined metadata
    combined_metadata = manager.create_combined_metadata(designer_metadata, augmented_metadata, reg_metadata)
    
    logger.info(f"✅ Personalized LoRA dataset setup complete!")
    logger.info(f"📁 Output directory: {args.output_dir}")
    logger.info(f"👨‍🎨 Designer samples: {len(designer_metadata)}")
    logger.info(f"🎨 Personalized samples: {len(augmented_metadata)}")
    logger.info(f"📊 Regularization samples: {len(reg_metadata)}")
    logger.info(f"🌐 Designer webpages: {args.output_dir}/designer_webpages/index.html")

if __name__ == "__main__":
    main() 