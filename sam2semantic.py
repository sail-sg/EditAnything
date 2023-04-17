# Edit Anything trained with Stable Diffusion + ControlNet + SAM  + BLIP2
# pip install mmcv

from torchvision.utils import save_image
from PIL import Image
import subprocess
from collections import OrderedDict
import numpy as np
import cv2
import textwrap
import torch
import os
from annotator.util import resize_image, HWC3
import mmcv
import random

# device = "cuda" if torch.cuda.is_available() else "cpu" # > 15GB GPU memory required
device = "cpu"
use_blip = True
use_gradio = True

if device == 'cpu':
    data_type = torch.float32
else:
    data_type = torch.float16
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
        "Salesforce/blip2-opt-2.7b", torch_dtype=data_type)


def region_classify_w_blip2(image):
    inputs = processor(image, return_tensors="pt").to(device, data_type)
    generated_ids = blip_model.generate(**inputs, max_new_tokens=15)
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

def region_level_semantic_api(image, topk=5):
    """
    rank regions by area, and classify each region with blip2
    Args:
        image: numpy array
        topk: int
    Returns:
        topk_region_w_class_label: list of dict with key 'class_label'
    """
    topk_region_w_class_label = []
    anns = mask_generator.generate(image)
    if len(anns) == 0:
        return []
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    for i in range(min(topk, len(sorted_anns))):
        ann = anns[i]
        m = ann['segmentation']
        m_3c = m[:,:, np.newaxis]
        m_3c = np.concatenate((m_3c,m_3c,m_3c), axis=2)
        bbox = ann['bbox']
        region = mmcv.imcrop(image*m_3c, np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]), scale=1)
        region_class_label = region_classify_w_blip2(region)
        ann['class_label'] = region_class_label
        print(ann['class_label'], str(bbox))
        topk_region_w_class_label.append(ann)
    return topk_region_w_class_label

def show_semantic_image_label(anns):
    """
    show semantic image label for each region
    Args:
        anns: list of dict with key 'class_label'
    Returns:
        full_img: numpy array
    """
    full_img = None
    # generate mask image
    for i in range(len(anns)):
        m = anns[i]['segmentation']
        if full_img is None:
            full_img = np.zeros((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        full_img[m != 0] = color_mask
    full_img = full_img*255
    # add text on this mask image
    for i in range(len(anns)):
        m = anns[i]['segmentation']
        class_label = anns[i]['class_label']
        # add text to region
        # Calculate the centroid of the region to place the text
        y, x = np.where(m != 0)
        x_center, y_center = int(np.mean(x)), int(np.mean(y))

        # Split the text into multiple lines
        max_width = 20  # Adjust this value based on your preferred maximum width
        wrapped_text = textwrap.wrap(class_label, width=max_width)

        # Add text to region
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 2
        font_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # red
        line_spacing = 40  # Adjust this value based on your preferred line

        for idx, line in enumerate(wrapped_text):
            y_offset = y_center - (len(wrapped_text) - 1) * line_spacing // 2 + idx * line_spacing
            text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            x_offset = x_center - text_size[0] // 2
            # Draw the text multiple times with small offsets to create a bolder appearance
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            for off_x, off_y in offsets:
                cv2.putText(full_img, line, (x_offset + off_x, y_offset + off_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    return full_img



image_path = "images/sa_224577.jpg"
input_image = Image.open(image_path)
detect_resolution=1024
input_image = resize_image(np.array(input_image, dtype=np.uint8), detect_resolution)
region_level_annots = region_level_semantic_api(input_image, topk=5)
output = show_semantic_image_label(region_level_annots)

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

save_image(image_list, "images/sample_semantic.jpg", nrow=2,
        normalize=True)

