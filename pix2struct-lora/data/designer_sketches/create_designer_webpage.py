#!/usr/bin/env python3
"""
Create Designer Webpage
This script creates a beautiful HTML webpage showcasing the 10 designer sketches and their code.
"""

import json
import os
from pathlib import Path

def create_designer_webpage():
    """Create HTML webpage for designer samples"""
    
    designer_dir = Path(".")  # Current directory (we're inside designer_sketches)
    output_file = designer_dir / "designer_showcase.html"
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎨 Designer Style Reference - LoRA Personalization</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            padding: 40px 20px;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .header p {
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }
        .stats-bar {
            background-color: #f8f9fa;
            padding: 20px;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            border-bottom: 1px solid #dee2e6;
        }
        .stat-item {
            text-align: center;
            margin: 10px;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #6c757d;
            font-size: 0.9em;
        }
        .gallery {
            padding: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 30px;
        }
        .sample-card {
            border: 1px solid #dee2e6;
            border-radius: 10px;
            overflow: hidden;
            transition: all 0.3s ease;
            background-color: white;
        }
        .sample-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        .card-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: center;
            font-weight: bold;
            font-size: 1.1em;
        }
        .card-images {
            display: flex;
            height: 200px;
        }
        .card-image {
            flex: 1;
            background-size: contain;
            background-position: center;
            background-repeat: no-repeat;
            background-color: #f8f9fa;
            position: relative;
            border-right: 1px solid #dee2e6;
        }
        .card-image:last-child {
            border-right: none;
        }
        .image-label {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(transparent, rgba(0,0,0,0.8));
            color: white;
            padding: 10px 5px 5px 5px;
            text-align: center;
            font-size: 0.8em;
            font-weight: bold;
        }
        .card-code {
            padding: 15px;
            background-color: #f8f9fa;
            border-top: 1px solid #dee2e6;
        }
        .code-info {
            font-size: 0.9em;
            color: #6c757d;
            margin-bottom: 10px;
        }
        .code-preview {
            background-color: #2d3748;
            color: #e2e8f0;
            padding: 10px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 0.8em;
            max-height: 100px;
            overflow-y: auto;
            margin-bottom: 10px;
        }
        .view-details {
            text-align: center;
            margin-top: 15px;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            font-size: 0.9em;
            transition: opacity 0.3s ease;
        }
        .btn:hover {
            opacity: 0.8;
        }
        .footer {
            background-color: #2d3748;
            color: white;
            text-align: center;
            padding: 30px;
        }
        .footer h3 {
            color: #667eea;
        }
        .usage-code {
            background-color: #1a202c;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 Designer Style Reference</h1>
            <p>LoRA Personalization Dataset - 10 Reference Sketches</p>
        </div>
        
        <div class="stats-bar">
            <div class="stat-item">
                <div class="stat-value">10</div>
                <div class="stat-label">Designer Samples</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">~"""
    
    # Count total components and characters
    total_components = 0
    total_html_chars = 0
    total_css_chars = 0
    
    for i in range(10):
        code_file = designer_dir / "code" / f"designer_{i}.json"
        if code_file.exists():
            try:
                with open(code_file, 'r') as f:
                    code_data = json.load(f)
                total_html_chars += len(code_data.get('html', ''))
                total_css_chars += len(code_data.get('css', ''))
                total_components += len(code_data.get('annotations', []))
            except:
                pass
    
    html_content += f"""{total_components}</div>
                <div class="stat-label">UI Components</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{total_html_chars:,}</div>
                <div class="stat-label">HTML Characters</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{total_css_chars:,}</div>
                <div class="stat-label">CSS Characters</div>
            </div>
        </div>
        
        <div class="gallery">"""
    
    # Add sample cards
    for i in range(10):
        sketch_file = designer_dir / "sketches" / f"designer_{i}.png"
        image_file = designer_dir / "images" / f"designer_{i}.png"
        code_file = designer_dir / "code" / f"designer_{i}.json"
        
        if not all([sketch_file.exists(), image_file.exists(), code_file.exists()]):
            continue
        
        try:
            with open(code_file, 'r') as f:
                code_data = json.load(f)
            
            html_preview = code_data.get('html', '')[:100] + '...' if len(code_data.get('html', '')) > 100 else code_data.get('html', '')
            css_preview = code_data.get('css', '')[:50] + '...' if len(code_data.get('css', '')) > 50 else code_data.get('css', '')
            
            html_content += f"""
            <div class="sample-card">
                <div class="card-header">
                    Designer Sample {i} (Original ID: sample_{[0,1,10,100,101,102,103,104,105,106][i]})
                </div>
                <div class="card-images">
                    <div class="card-image" style="background-image: url('images/designer_{i}.png');">
                        <div class="image-label">Website Screenshot</div>
                    </div>
                    <div class="card-image" style="background-image: url('sketches/designer_{i}.png');">
                        <div class="image-label">Designer Sketch</div>
                    </div>
                </div>
                <div class="card-code">
                    <div class="code-info">
                        <strong>HTML:</strong> {len(code_data.get('html', ''))} chars | 
                        <strong>CSS:</strong> {len(code_data.get('css', ''))} chars | 
                        <strong>Components:</strong> {len(code_data.get('annotations', []))}
                    </div>
                    <div class="code-preview">
                        &lt;html&gt;{html_preview}&lt;/html&gt;
                        &lt;style&gt;{css_preview}&lt;/style&gt;
                    </div>
                    <div class="view-details">
                        <button class="btn" onclick="viewDetails({i})">View Full Code</button>
                    </div>
                </div>
            </div>"""
        except:
            continue
    
    html_content += """
        </div>
        
        <div class="footer">
            <h3>🚀 Next Steps: LoRA Training</h3>
            <p>These 10 designer samples will be used to:</p>
            <ul style="text-align: left; display: inline-block; margin: 20px 0;">
                <li>Extract UI components representing the designer's style</li>
                <li>Generate synthetic training data using component substitution</li>
                <li>Train a LoRA adapter to personalize Pix2Struct to this style</li>
                <li>Evaluate style consistency and generation quality</li>
            </ul>
            
            <h4>🔧 Generate Synthetic Data:</h4>
            <div class="usage-code">
python generate_synthetic_designer_data.py --num_samples 50
            </div>
            
            <h4>🎯 Train LoRA Adapter:</h4>
            <div class="usage-code">
python train_personalized_lora.py
            </div>
            
            <p style="margin-top: 30px; opacity: 0.7;">
                Generated on: """ + str(Path.cwd()) + """<br>
                For LoRA-based sketch-to-code personalization
            </p>
        </div>
    </div>
    
    <script>
        function viewDetails(sampleId) {
            // In a real implementation, this would open a modal or navigate to a detail page
            alert(`Would show full code details for Designer Sample ${sampleId}`);
        }
    </script>
</body>
</html>
"""
    
    # Save the webpage
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Designer webpage created: {output_file}")
    print(f"🌐 Open in browser: file://{output_file.absolute()}")

if __name__ == "__main__":
    create_designer_webpage() 