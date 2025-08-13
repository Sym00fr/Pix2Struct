from PIL import Image
from transformers import Pix2StructForConditionalGeneration, AutoProcessor
import torch

repo_id = "google/pix2struct-base"

processor = AutoProcessor.from_pretrained(repo_id)
model = Pix2StructForConditionalGeneration.from_pretrained(repo_id, is_encoder_decoder=True)