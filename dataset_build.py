from PIL import Image
import json

from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import os

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def get_blip2_text(image):
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text


data_path = "files"
save_path = ""

image_names = os.listdir(data_path)
image_names = sorted(image_names)

text_data = {}
f = open("data.txt","w")
for each in image_names:
    if '.jpg' in each:
        this_data = {}
        this_data['target'] = each
        this_data['source'] = each[:-4]+'.json'
        this_image = Image.open(os.path.join(data_path, each))
        print(each)
        generated_text = get_blip2_text(this_image)
        this_data['prompt'] = generated_text
        print(this_data)
        f.write(str(this_data)+"\n")
f.close()

        



