# Edit Anything trained with Stable Diffusion + ControlNet + SAM  + BLIP2
from torchvision.utils import save_image
from PIL import Image
from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from pytorch_lightning import seed_everything
from share import *
import config

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

device = "cuda" if torch.cuda.is_available() else "cpu"
use_blip = False
use_gradio = False

# Diffusion init using diffusers.

from vlpart.vlpart import build_vlpart
from segment_anything import build_sam, SamPredictor
from segment_anything.utils.amg import remove_small_regions
import detectron2.data.transforms as T

# diffusers==0.14.0 required.
from diffusers import ControlNetModel, UniPCMultistepScheduler
from utils.stable_diffusion_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline
from diffusers.utils import load_image
import torch

base_model_path = "stabilityai/stable-diffusion-2-inpainting"
controlnet_path = "shgao/edit-anything-v0-1-1"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16
)
# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed
pipe.enable_xformers_memory_efficient_attention()

# pipe.enable_model_cpu_offload() # disable for now because of unknow bug in accelerate
pipe.to(device)

# Segment-Anything init.
# pip install git+https://github.com/facebookresearch/segment-anything.git
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
vlpart_checkpoint = "./models/swinbase_part_0a0000.pth"
model_type = "default"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
vlpart = build_vlpart(checkpoint=vlpart_checkpoint)
vlpart.to(device=device)
sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device=device))

# BLIP2 init.
if use_blip:
    # need the latest transformers
    # pip install git+https://github.com/huggingface/transformers.git
    from transformers import AutoProcessor, Blip2ForConditionalGeneration

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    blip_model.to(device)
    blip_model.to(device)


def get_blip2_text(image):
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    generated_ids = blip_model.generate(**inputs, max_new_tokens=50)
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    full_img = None

    # for ann in sorted_anns:
    for i in range(len(sorted_anns)):
        ann = anns[i]
        m = ann['segmentation']
        if full_img is None:
            full_img = np.zeros((m.shape[0], m.shape[1], 3))
            map = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint16)
        map[m != 0] = i + 1
        color_mask = np.random.random((1, 3)).tolist()[0]
        full_img[m != 0] = color_mask
    full_img = full_img * 255
    # anno encoding from https://github.com/LUSSeg/ImageNet-S
    res = np.zeros((map.shape[0], map.shape[1], 3))
    res[:, :, 0] = map % 256
    res[:, :, 1] = map // 256
    res.astype(np.float32)
    return full_img, res


def get_sam_control(image):
    masks = mask_generator.generate(image)
    full_img, res = show_anns(masks)
    return full_img, res


def prompt2mask(original_image, text_prompt):
    # original_image = original_image[:, :, :3]
    preprocess = T.ResizeShortestEdge([800, 800], 1333)
    height, width = original_image.shape[:2]
    image = preprocess.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = {"image": image, "height": height, "width": width}
    with torch.no_grad():
        predictions = vlpart.inference([inputs], text_prompt=text_prompt)[0]
    boxes, masks = None, None
    filter_scores, filter_boxes, filter_classes = [], [], []

    if "instances" in predictions:
        instances = predictions['instances'].to('cpu')
        boxes = instances.pred_boxes.tensor if instances.has("pred_boxes") else None
        scores = instances.scores if instances.has("scores") else None
        classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None

        num_obj = len(scores)
        for obj_ind in range(num_obj):
            category_score = scores[obj_ind]
            if category_score < 0.7:
                continue
            filter_scores.append(category_score)
            filter_boxes.append(boxes[obj_ind])
            filter_classes.append(classes[obj_ind])

    if len(filter_boxes) > 0:
        # sam model inference
        sam_predictor.set_image(original_image)

        boxes_filter = torch.stack(filter_boxes)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filter, original_image.shape[:2])
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(device),
            multimask_output=False,
        )

        # remove small disconnected regions and holes
        fine_masks = []
        for mask in masks.to('cpu').numpy():  # masks: [num_masks, 1, h, w]
            fine_masks.append(remove_small_regions(mask[0], 400, mode="holes")[0])
        masks = np.stack(fine_masks, axis=0)[:, np.newaxis]
        masks = torch.from_numpy(masks)

    num_obj = len(scores)
    final_m = torch.zeros((original_image.shape[0], original_image.shape[1]))
    for obj_ind in range(num_obj):
        # box = boxes[obj_ind]
        score = scores[obj_ind]
        if score < 0.5:
            continue
        m = masks[obj_ind][0]
        final_m += m
    final_m = (final_m > 0).to('cpu').numpy()
    # print(final_m.max(), final_m.min())
    return np.dstack((final_m, final_m, final_m)) * 255


def process(input_image, mask_prompt, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution,
            ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        mask_image = np.array(prompt2mask(input_image, mask_prompt), dtype=np.uint8)

        if use_blip:
            print("Generating text:")
            blip2_prompt = get_blip2_text(input_image)
            print("Generated text:", blip2_prompt)
            if len(prompt) > 0:
                prompt = blip2_prompt + ',' + prompt
            else:
                prompt = blip2_prompt
            print("All text:", prompt)

        input_image = HWC3(input_image)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        print("Generating SAM seg:")
        # the default SAM model is trained with 1024 size.
        full_segmask, detected_map = get_sam_control(
            resize_image(input_image, detect_resolution))

        detected_map = HWC3(detected_map.astype(np.uint8))
        detected_map = cv2.resize(
            detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(
            detected_map.copy()).float().cuda()
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        mask_image = HWC3(mask_image.astype(np.uint8))
        mask_image = cv2.resize(
            mask_image, (W, H), interpolation=cv2.INTER_LINEAR)
        mask_image = Image.fromarray(mask_image)

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)
        generator = torch.manual_seed(seed)
        x_samples = pipe(
            image=img,
            mask_image=mask_image,
            prompt=[prompt + ', ' + a_prompt] * num_samples,
            negative_prompt=[n_prompt] * num_samples,
            num_images_per_prompt=num_samples,
            num_inference_steps=ddim_steps,
            generator=generator,
            controlnet_conditioning_image=control.type(torch.float16),
            height=H,
            width=W,
        ).images

        results = [x_samples[i] for i in range(num_samples)]
    return [full_segmask, mask_image] + results


def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


# disable gradio when not using GUI.
if not use_gradio:
    image_path = "assets/dog.png"
    input_image = Image.open(image_path)

    input_image = np.array(input_image, dtype=np.uint8)[:, :, :3]

    mask_prompt = 'dog head.'
    # mask_image = np.array(prompt2mask(input_image, mask_prompt), dtype=np.uint8)
    prompt = "cute dogs"
    a_prompt = 'best quality, extremely detailed'
    n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    num_samples = 3
    image_resolution = 512
    detect_resolution = 512
    ddim_steps = 30
    guess_mode = False
    strength = 1.0
    scale = 9.0
    seed = -1
    eta = 0.0

    outputs = process(input_image, mask_prompt, prompt, a_prompt, n_prompt, num_samples, image_resolution,
                      detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)

    image_list = []
    input_image = resize_image(input_image, 512)
    image_list.append(torch.tensor(input_image))
    for i in range(len(outputs)):
        each = outputs[i]
        if type(each) is not np.ndarray:
            each = np.array(each, dtype=np.uint8)
        each = resize_image(each, 512)
        print(i, each.shape)
        image_list.append(torch.tensor(each))

    image_list = torch.stack(image_list).permute(0, 3, 1, 2)

    save_image(image_list, "sample.jpg", nrow=3,
               normalize=True, value_range=(0, 255))
else:
    print("The GUI is not tested yet. Please open an issue if you find bugs.")
    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            gr.Markdown(
                "## Edit Anything powered by ControlNet+SAM+BLIP2+Stable Diffusion")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type="numpy")
                prompt = gr.Textbox(label="Prompt")
                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    num_samples = gr.Slider(
                        label="Images", minimum=1, maximum=12, value=1, step=1)
                    image_resolution = gr.Slider(
                        label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                    strength = gr.Slider(
                        label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                    guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                    detect_resolution = gr.Slider(
                        label="SAM Resolution", minimum=128, maximum=2048, value=1024, step=1)
                    ddim_steps = gr.Slider(
                        label="Steps", minimum=1, maximum=100, value=20, step=1)
                    scale = gr.Slider(
                        label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                    seed = gr.Slider(label="Seed", minimum=-1,
                                     maximum=2147483647, step=1, randomize=True)
                    eta = gr.Number(label="eta (DDIM)", value=0.0)
                    mask_prompt = gr.Textbox(
                        label="Mask Prompt", value='best quality, extremely detailed')
                    a_prompt = gr.Textbox(
                        label="Added Prompt", value='best quality, extremely detailed')
                    n_prompt = gr.Textbox(label="Negative Prompt",
                                          value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
            with gr.Column():
                result_gallery = gr.Gallery(
                    label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
        ips = [input_image, mask_prompt, prompt, a_prompt, n_prompt, num_samples, image_resolution,
               detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
        run_button.click(fn=process, inputs=ips, outputs=[result_gallery])

    block.launch(server_name='0.0.0.0')
