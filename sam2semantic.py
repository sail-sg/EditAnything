# Edit Anything trained with Stable Diffusion + ControlNet + SAM  + BLIP2
from torchvision.utils import save_image
from PIL import Image
from pytorch_lightning import seed_everything
import subprocess
from collections import OrderedDict

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os
import requests
from io import BytesIO
from annotator.util import resize_image, HWC3
import mmcv

device = "cuda" if torch.cuda.is_available() else "cpu"
use_blip = True
use_gradio = True

# Diffusion init using diffusers.

# diffusers==0.14.0 required.
from diffusers.utils import load_image

base_model_path = "stabilityai/stable-diffusion-2-inpainting"
config_dict = OrderedDict([('SAM Pretrained(v0-1): Good Natural Sense', 'shgao/edit-anything-v0-1-1'),
                        ('LAION Pretrained(v0-3): Good Face', 'shgao/edit-anything-v0-3'),
                        ('SD Inpainting: Not keep position', 'stabilityai/stable-diffusion-2-inpainting')
                        ])


# Segment-Anything init.
# pip install git+https://github.com/facebookresearch/segment-anything.git
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    print('segment_anything not installed')
    result = subprocess.run(['pip', 'install', 'git+https://github.com/facebookresearch/segment-anything.git'], check=True)
    print(f'Install segment_anything {result}')   
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
if not os.path.exists('./models/sam_vit_h_4b8939.pth'):
    result = subprocess.run(['wget', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth', '-P', 'models'], check=True)
    print(f'Download sam_vit_h_4b8939.pth {result}')   
sam_checkpoint = "models/sam_vit_h_4b8939.pth"
model_type = "default"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)


# BLIP2 init.
if use_blip:
    # need the latest transformers
    # pip install git+https://github.com/huggingface/transformers.git
    from transformers import AutoProcessor, Blip2ForConditionalGeneration

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto")


def get_blip2_text(image):
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    generated_ids = blip_model.generate(**inputs, max_new_tokens=15)
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text


def show_anns_label(image, topk=8):
    anns = mask_generator.generate(image)
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    full_img = None

    # for ann in sorted_anns:
    if topk>len(sorted_anns):
        topk = len(sorted_anns)
    for i in range(topk):
        ann = anns[i]
        m = ann['segmentation']
        m_3c = m[:,:, np.newaxis]
        m_3c = np.concatenate((m_3c,m_3c,m_3c),axis=2)
        bbox = ann['bbox']
        patch_large = mmcv.imcrop(image*m_3c, np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]), scale=1)
        this_text = get_blip2_text(patch_large)
        print(this_text)
        if full_img is None:
            full_img = np.zeros((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        full_img[m != 0] = color_mask
    full_img = full_img*255
    # full_img = Image.fromarray(np.uint8(full_img))
    return full_img




image_path = "../data/files/sa_223758.jpg"
input_image = Image.open(image_path)
input_image = np.array(input_image, dtype=np.uint8)
detect_resolution=1024
output = show_anns_label(resize_image(input_image, detect_resolution))


image_list = []
input_image = resize_image(input_image, 512)
output = resize_image(output, 512)
input_image = np.array(input_image, dtype=np.uint8)
output = np.array(output, dtype=np.uint8)
image_list.append(torch.tensor(input_image).float())
image_list.append(torch.tensor(output).float())
for each in image_list:
    print(each.shape, type(each))
    print(each.max(), each.min())


image_list = torch.stack(image_list).permute(0, 3, 1, 2)
print(image_list.shape)

save_image(image_list, "sample.jpg", nrow=2,
        normalize=True)

