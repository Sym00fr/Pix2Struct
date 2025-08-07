from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from peft import PeftModel
from PIL import Image

# Carica l'immagine da file
sketch_image = Image.open("Sketch_to_convert.png").convert("RGB")

# Load base model and processor
base_model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-large")
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-large")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./outputs/final_lora_adapter")

# Generate code from sketch
encoding = processor(images=sketch_image, text="Generate website code", return_tensors="pt")
outputs = model.generate(**encoding, max_length=512)
generated_code = processor.decode(outputs[0], skip_special_tokens=True)
print('ok')
print(generated_code)