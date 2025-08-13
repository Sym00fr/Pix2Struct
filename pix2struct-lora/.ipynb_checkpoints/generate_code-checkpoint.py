'''from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from peft import PeftModel
from PIL import Image
import json

# Carica l'immagine da file
sketch_image = Image.open("Sketch_to_convert.png")#convert("RGB")

# Load base model and processor
base_model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-large")
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-large")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./outputs/final_lora_adapter")

# Generate code from sketch
encoding = processor(images=sketch_image, text="Generate website code", return_tensors="pt")
outputs = model.generate(**encoding, max_length=512)
#outputs = base_model.generate(**encoding, max_length=512)
generated_code = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
print('ok')

#with open("outputs/evaluation_results_prova.json", 'w') as f:
#    json.dump(generated_code, f, indent=2, default=str)

print(generated_code)'''

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor, AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image
import json
import torch

print('ok import')

# Carica l'immagine da file
sketch_image = Image.open("./Sketch_to_convert.png").convert("RGB")
print(isinstance(sketch_image, Image.Image))

print('ok load image')

# Load base model and processor
base_model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-base")
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-base")

print('ok load model an processor')

checkpoint = torch.load("Pix2Struct_SketchSynthBootstrap_Complete_epoch[9].pth", map_location='cpu')
base_model.load_state_dict(checkpoint['model_state_dict'])

print('ok load checkpoint')

#processor = AutoProcessor.from_pretrained("google/pix2struct-large")
#model = AutoModelForVision2Seq.from_pretrained("google/pix2struct-large")

# Load LoRA adapter
#model = PeftModel.from_pretrained(base_model, "./outputs/final_lora_adapter")

# Generate code from sketch
encoding = processor(images=sketch_image, text="Generate website code", return_tensors="pt")

print('ok encoding')
outputs = base_model.generate(**encoding, max_length=4096)

print('ok outputs')
generated_code = processor.decode(outputs[0], skip_special_tokens=True)

print(generated_code)

'''from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image
from peft import PeftModel
import torch

processor = Pix2StructProcessor.from_pretrained("google/pix2struct-large")
        
base_model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-large")
        
lora_model = PeftModel.from_pretrained(base_model, "outputs/final_lora_adapter")

sketch_image = Image.open("Sketch_to_convert.png").convert('RGB')

model_type='lora'


        
# Resize image
original_width, original_height = sketch_image.size
max_size = 384
        
if max(original_width, original_height) > max_size:
    scale_factor = max_size / max(original_width, original_height)
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    sketch_image = sketch_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
# Process input
encoding = processor(
    images=sketch_image,
    text="Generate website code from this sketch",
    return_tensors="pt",
    max_patches=96
)
        
# Generate
model = lora_model if model_type == "lora" else base_model
        
with torch.no_grad():
    generated_ids = model.generate(
        flattened_patches=encoding['flattened_patches'],
        attention_mask=encoding['attention_mask'],
        max_length=512,
        num_beams=4,
        do_sample=False,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id
    )
        
# Decode
generated_text = processor.tokenizer.decode(
    generated_ids[0], skip_special_tokens=True
)
        

print(generated_text)'''



