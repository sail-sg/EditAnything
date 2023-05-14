# Edit Anything trained with Stable Diffusion + ControlNet + SAM  + BLIP2
from torchvision.utils import save_image
from PIL import Image
from pytorch_lightning import seed_everything
import subprocess
from collections import OrderedDict
import re
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os
import requests
from io import BytesIO
from annotator.util import resize_image, HWC3, resize_points

import torch
from safetensors.torch import load_file
from collections import defaultdict
from diffusers import StableDiffusionControlNetPipeline
from diffusers import ControlNetModel, UniPCMultistepScheduler
from utils.stable_diffusion_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline
# need the latest transformers
# pip install git+https://github.com/huggingface/transformers.git
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from diffusers import ControlNetModel, DiffusionPipeline
import PIL.Image

# Segment-Anything init.
# pip install git+https://github.com/facebookresearch/segment-anything.git
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
except ImportError:
    print('segment_anything not installed')
    result = subprocess.run(
        ['pip', 'install', 'git+https://github.com/facebookresearch/segment-anything.git'], check=True)
    print(f'Install segment_anything {result}')
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
if not os.path.exists('./models/sam_vit_h_4b8939.pth'):
    result = subprocess.run(
        ['wget', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth', '-P', 'models'], check=True)
    print(f'Download sam_vit_h_4b8939.pth {result}')

device = "cuda" if torch.cuda.is_available() else "cpu"

config_dict = OrderedDict([
    ('LAION Pretrained(v0-4)-SD15', 'shgao/edit-anything-v0-4-sd15'),
    ('LAION Pretrained(v0-4)-SD21', 'shgao/edit-anything-v0-4-sd21'),
])


def init_sam_model(sam_generator=None, mask_predictor=None):
    if sam_generator is not None and mask_predictor is not None:
        return sam_generator, mask_predictor
    sam_checkpoint = "models/sam_vit_h_4b8939.pth"
    model_type = "default"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_generator = SamAutomaticMaskGenerator(
        sam) if sam_generator is None else sam_generator
    mask_predictor = SamPredictor(
        sam) if mask_predictor is None else mask_predictor
    return sam_generator, mask_predictor


def init_blip_processor():
    blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    return blip_processor


def init_blip_model():
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto")
    return blip_model


def get_pipeline_embeds(pipeline, prompt, negative_prompt, device):
    # https://github.com/huggingface/diffusers/issues/2136
    """ Get pipeline embeds for prompts bigger than the maxlength of the pipe
    :param pipeline:
    :param prompt:
    :param negative_prompt:
    :param device:
    :return:
    """
    max_length = pipeline.tokenizer.model_max_length

    # simple way to determine length of tokens
    count_prompt = len(re.split(r', ', prompt))
    count_negative_prompt = len(re.split(r', ', negative_prompt))

    # create the tensor based on which prompt is longer
    if count_prompt >= count_negative_prompt:
        input_ids = pipeline.tokenizer(
            prompt, return_tensors="pt", truncation=False).input_ids.to(device)
        shape_max_length = input_ids.shape[-1]
        negative_ids = pipeline.tokenizer(negative_prompt, truncation=False, padding="max_length",
                                          max_length=shape_max_length, return_tensors="pt").input_ids.to(device)
    else:
        negative_ids = pipeline.tokenizer(
            negative_prompt, return_tensors="pt", truncation=False).input_ids.to(device)
        shape_max_length = negative_ids.shape[-1]
        input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False, padding="max_length",
                                       max_length=shape_max_length).input_ids.to(device)

    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(pipeline.text_encoder(
            input_ids[:, i: i + max_length])[0])
        neg_embeds.append(pipeline.text_encoder(
            negative_ids[:, i: i + max_length])[0])

    return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)


def load_lora_weights(pipeline, checkpoint_path, multiplier, device, dtype):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    # load LoRA weight from .safetensors
    if isinstance(checkpoint_path, str):

        state_dict = load_file(checkpoint_path, device=device)

        updates = defaultdict(dict)
        for key, value in state_dict.items():
            # it is suggested to print out the key, it usually will be something like below
            # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

            layer, elem = key.split('.', 1)
            updates[layer][elem] = value

        # directly update weight in diffusers model
        for layer, elems in updates.items():

            if "text" in layer:
                layer_infos = layer.split(
                    LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                curr_layer = pipeline.text_encoder
            else:
                layer_infos = layer.split(
                    LORA_PREFIX_UNET + "_")[-1].split("_")
                curr_layer = pipeline.unet

            # find the target layer
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        temp_name += "_" + layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)

            # get elements for this layer
            weight_up = elems['lora_up.weight'].to(dtype)
            weight_down = elems['lora_down.weight'].to(dtype)
            alpha = elems['alpha']
            if alpha:
                alpha = alpha.item() / weight_up.shape[1]
            else:
                alpha = 1.0

            # update weight
            if len(weight_up.shape) == 4:
                curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(
                    3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            else:
                curr_layer.weight.data += multiplier * \
                    alpha * torch.mm(weight_up, weight_down)
    else:
        for ckptpath in checkpoint_path:
            state_dict = load_file(ckptpath, device=device)

            updates = defaultdict(dict)
            for key, value in state_dict.items():
                # it is suggested to print out the key, it usually will be something like below
                # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

                layer, elem = key.split('.', 1)
                updates[layer][elem] = value

            # directly update weight in diffusers model
            for layer, elems in updates.items():
                if "text" in layer:
                    layer_infos = layer.split(
                        LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                    curr_layer = pipeline.text_encoder
                else:
                    layer_infos = layer.split(
                        LORA_PREFIX_UNET + "_")[-1].split("_")
                    curr_layer = pipeline.unet

                # find the target layer
                temp_name = layer_infos.pop(0)
                while len(layer_infos) > -1:
                    try:
                        curr_layer = curr_layer.__getattr__(temp_name)
                        if len(layer_infos) > 0:
                            temp_name = layer_infos.pop(0)
                        elif len(layer_infos) == 0:
                            break
                    except Exception:
                        if len(temp_name) > 0:
                            temp_name += "_" + layer_infos.pop(0)
                        else:
                            temp_name = layer_infos.pop(0)

                # get elements for this layer
                weight_up = elems['lora_up.weight'].to(dtype)
                weight_down = elems['lora_down.weight'].to(dtype)
                alpha = elems['alpha']
                if alpha:
                    alpha = alpha.item() / weight_up.shape[1]
                else:
                    alpha = 1.0

                # update weight
                if len(weight_up.shape) == 4:
                    curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(
                        3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                else:
                    curr_layer.weight.data += multiplier * \
                        alpha * torch.mm(weight_up, weight_down)
    return pipeline


def make_inpaint_condition(image, image_mask):
    # image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image = image / 255.0
    print("img", image.max(), image.min(), image_mask.max(), image_mask.min())
    # image_mask = np.array(image_mask.convert("L"))
    assert image.shape[0:1] == image_mask.shape[0:
                                                1], "image and image_mask must have the same image size"
    image[image_mask > 128] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def obtain_generation_model(base_model_path, lora_model_path, controlnet_path, generation_only=False, extra_inpaint=True, lora_weight=1.0):
    controlnet = []
    controlnet.append(ControlNetModel.from_pretrained(
        controlnet_path, torch_dtype=torch.float16))  # sam control
    if (not generation_only) and extra_inpaint:  # inpainting control
        print("Warning: ControlNet based inpainting model only support SD1.5 for now.")
        controlnet.append(
            ControlNetModel.from_pretrained(
                'lllyasviel/control_v11p_sd15_inpaint', torch_dtype=torch.float16)  # inpainting controlnet
        )

    if generation_only:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_path, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
        )
    else:
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            base_model_path, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
        )
    if lora_model_path is not None:
        pipe = load_lora_weights(
            pipe, [lora_model_path], lora_weight, 'cpu', torch.float32)
    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config)
    # remove following line if xformers is not installed
    pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_model_cpu_offload()
    return pipe


def obtain_tile_model(base_model_path, lora_model_path, lora_weight=1.0):
    controlnet = ControlNetModel.from_pretrained(
        'lllyasviel/control_v11f1e_sd15_tile', torch_dtype=torch.float16)  # tile controlnet
    if base_model_path == 'runwayml/stable-diffusion-v1-5' or base_model_path == 'stabilityai/stable-diffusion-2-inpainting':
        print("base_model_path", base_model_path)
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
        )
    else:
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            base_model_path, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
        )
    if lora_model_path is not None:
        pipe = load_lora_weights(
            pipe, [lora_model_path], lora_weight, 'cpu', torch.float32)
    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config)
    # remove following line if xformers is not installed
    pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_model_cpu_offload()
    return pipe


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
    full_img = full_img*255
    # anno encoding from https://github.com/LUSSeg/ImageNet-S
    res = np.zeros((map.shape[0], map.shape[1], 3))
    res[:, :, 0] = map % 256
    res[:, :, 1] = map // 256
    res.astype(np.float32)
    full_img = Image.fromarray(np.uint8(full_img))
    return full_img, res


class EditAnythingLoraModel:
    def __init__(self,
                 base_model_path='../chilloutmix_NiPrunedFp32Fix',
                 lora_model_path='../40806/mix4', use_blip=True,
                 blip_processor=None,
                 blip_model=None,
                 sam_generator=None,
                 controlmodel_name='LAION Pretrained(v0-4)-SD15',
                 # used when the base model is not an inpainting model.
                 extra_inpaint=True,
                 tile_model=None,
                 lora_weight=1.0,
                 mask_predictor=None
                 ):
        self.device = device
        self.use_blip = use_blip

        # Diffusion init using diffusers.
        self.default_controlnet_path = config_dict[controlmodel_name]
        self.base_model_path = base_model_path
        self.lora_model_path = lora_model_path
        self.defalut_enable_all_generate = False
        self.extra_inpaint = extra_inpaint
        self.pipe = obtain_generation_model(
            base_model_path, lora_model_path, self.default_controlnet_path, generation_only=False, extra_inpaint=extra_inpaint, lora_weight=lora_weight)

        # Segment-Anything init.
        self.sam_generator, self.mask_predictor = init_sam_model(
            sam_generator, mask_predictor)
        # BLIP2 init.
        if use_blip:
            if blip_processor is not None:
                self.blip_processor = blip_processor
            else:
                self.blip_processor = init_blip_processor()

            if blip_model is not None:
                self.blip_model = blip_model
            else:
                self.blip_model = init_blip_model()

        # tile model init.
        if tile_model is not None:
            self.tile_pipe = tile_model
        else:
            self.tile_pipe = obtain_tile_model(
                base_model_path, lora_model_path, lora_weight=lora_weight)

    def get_blip2_text(self, image):
        inputs = self.blip_processor(image, return_tensors="pt").to(
            self.device, torch.float16)
        generated_ids = self.blip_model.generate(**inputs, max_new_tokens=50)
        generated_text = self.blip_processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text

    def get_sam_control(self, image):
        masks = self.sam_generator.generate(image)
        full_img, res = show_anns(masks)
        return full_img, res

    def get_click_mask(self, image, clicked_points):
        self.mask_predictor.set_image(image)
        # Separate the points and labels
        points, labels = zip(*[(point[:2], point[2])
                             for point in clicked_points])

        # Convert the points and labels to numpy arrays
        input_point = np.array(points)
        input_label = np.array(labels)

        masks, _, _ = self.mask_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        return masks

    @torch.inference_mode()
    def process_image_click(self, original_image: gr.Image,
                            point_prompt: gr.Radio,
                            clicked_points: gr.State,
                            image_resolution,
                            evt: gr.SelectData):
        # Get the clicked coordinates
        clicked_coords = evt.index
        x, y = clicked_coords
        label = point_prompt
        lab = 1 if label == "Foreground Point" else 0
        clicked_points.append((x, y, lab))

        input_image = np.array(original_image, dtype=np.uint8)
        H, W, C = input_image.shape
        input_image = HWC3(input_image)
        img = resize_image(input_image, image_resolution)

        # Update the clicked_points
        resized_points = resize_points(clicked_points,
                                       input_image.shape,
                                       image_resolution)
        mask_click_np = self.get_click_mask(img, resized_points)

        # Convert mask_click_np to HWC format
        mask_click_np = np.transpose(mask_click_np, (1, 2, 0)) * 255.0

        mask_image = HWC3(mask_click_np.astype(np.uint8))
        mask_image = cv2.resize(
            mask_image, (W, H), interpolation=cv2.INTER_LINEAR)
        # mask_image = Image.fromarray(mask_image_tmp)

        # Draw circles for all clicked points
        edited_image = input_image
        for x, y, lab in clicked_points:
            # Set the circle color based on the label
            color = (255, 0, 0) if lab == 1 else (0, 0, 255)

            # Draw the circle
            edited_image = cv2.circle(edited_image, (x, y), 20, color, -1)

        # Set the opacity for the mask_image and edited_image
        opacity_mask = 0.75
        opacity_edited = 1.0

        # Combine the edited_image and the mask_image using cv2.addWeighted()
        overlay_image = cv2.addWeighted(
            edited_image, opacity_edited,
            (mask_image * np.array([0/255, 255/255, 0/255])).astype(np.uint8),
            opacity_mask, 0
        )

        return Image.fromarray(overlay_image), clicked_points, Image.fromarray(mask_image)

    @torch.inference_mode()
    def process(self, source_image, enable_all_generate, mask_image,
                control_scale,
                enable_auto_prompt, a_prompt, n_prompt,
                num_samples, image_resolution, detect_resolution,
                ddim_steps, guess_mode, strength, scale, seed, eta,
                enable_tile=True, refine_alignment_ratio=None, condition_model=None):

        if condition_model is None:
            this_controlnet_path = self.default_controlnet_path
        else:
            this_controlnet_path = config_dict[condition_model]
        input_image = source_image["image"] if isinstance(
            source_image, dict) else np.array(source_image, dtype=np.uint8)
        if mask_image is None:
            if enable_all_generate != self.defalut_enable_all_generate:
                self.pipe = obtain_generation_model(
                    self.base_model_path, self.lora_model_path, this_controlnet_path, enable_all_generate, self.extra_inpaint)

                self.defalut_enable_all_generate = enable_all_generate
            if enable_all_generate:
                print("source_image",
                      source_image["mask"].shape, input_image.shape,)
                mask_image = np.ones(
                    (input_image.shape[0], input_image.shape[1], 3))*255
            else:
                mask_image = source_image["mask"]
        else:
            mask_image = np.array(mask_image, dtype=np.uint8)
        if self.default_controlnet_path != this_controlnet_path:
            print("To Use:", this_controlnet_path,
                  "Current:", self.default_controlnet_path)
            print("Change condition model to:", this_controlnet_path)
            self.pipe = obtain_generation_model(
                self.base_model_path, self.lora_model_path, this_controlnet_path, enable_all_generate, self.extra_inpaint)
            self.default_controlnet_path = this_controlnet_path
            torch.cuda.empty_cache()

        with torch.no_grad():
            if self.use_blip and enable_auto_prompt:
                print("Generating text:")
                blip2_prompt = self.get_blip2_text(input_image)
                print("Generated text:", blip2_prompt)
                if len(a_prompt) > 0:
                    a_prompt = blip2_prompt + ',' + a_prompt
                else:
                    a_prompt = blip2_prompt

            input_image = HWC3(input_image)

            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape

            print("Generating SAM seg:")
            # the default SAM model is trained with 1024 size.
            full_segmask, detected_map = self.get_sam_control(
                resize_image(input_image, detect_resolution))

            detected_map = HWC3(detected_map.astype(np.uint8))
            detected_map = cv2.resize(
                detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

            control = torch.from_numpy(
                detected_map.copy()).float().cuda()
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            mask_imag_ori = HWC3(mask_image.astype(np.uint8))
            mask_image_tmp = cv2.resize(
                mask_imag_ori, (W, H), interpolation=cv2.INTER_LINEAR)
            mask_image = Image.fromarray(mask_image_tmp)

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)
            generator = torch.manual_seed(seed)
            postive_prompt = a_prompt
            negative_prompt = n_prompt
            prompt_embeds, negative_prompt_embeds = get_pipeline_embeds(
                self.pipe, postive_prompt, negative_prompt, "cuda")
            prompt_embeds = torch.cat([prompt_embeds] * num_samples, dim=0)
            negative_prompt_embeds = torch.cat(
                [negative_prompt_embeds] * num_samples, dim=0)
            if enable_all_generate and not self.extra_inpaint:
                self.pipe.safety_checker = lambda images, clip_input: (
                    images, False)
                x_samples = self.pipe(
                    prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                    num_images_per_prompt=num_samples,
                    num_inference_steps=ddim_steps,
                    generator=generator,
                    height=H,
                    width=W,
                    image=[control.type(torch.float16)],
                    controlnet_conditioning_scale=[float(control_scale)],
                ).images
            else:
                multi_condition_image = []
                multi_condition_scale = []
                multi_condition_image.append(control.type(torch.float16))
                multi_condition_scale.append(float(control_scale))
                if self.extra_inpaint:
                    inpaint_image = make_inpaint_condition(img, mask_image_tmp)
                    print(inpaint_image.shape)
                    multi_condition_image.append(
                        inpaint_image.type(torch.float16))
                    multi_condition_scale.append(1.0)
                x_samples = self.pipe(
                    image=img,
                    mask_image=mask_image,
                    prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                    num_images_per_prompt=num_samples,
                    num_inference_steps=ddim_steps,
                    generator=generator,
                    controlnet_conditioning_image=multi_condition_image,
                    height=H,
                    width=W,
                    controlnet_conditioning_scale=multi_condition_scale,
                ).images
            results = [x_samples[i] for i in range(num_samples)]

            results_tile = []
            if enable_tile:
                prompt_embeds, negative_prompt_embeds = get_pipeline_embeds(
                    self.tile_pipe, postive_prompt, negative_prompt, "cuda")
                for i in range(num_samples):
                    img_tile = PIL.Image.fromarray(resize_image(np.array(x_samples[i]), 1024))
                    if i == 0:
                        mask_image_tile = cv2.resize(
                        mask_imag_ori, (img_tile.size[0], img_tile.size[1]), interpolation=cv2.INTER_LINEAR)
                        mask_image_tile = Image.fromarray(mask_image_tile)
                    x_samples_tile = self.tile_pipe(
                        image=img_tile,
                        mask_image=mask_image_tile,
                        prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                        num_images_per_prompt=1,
                        num_inference_steps=ddim_steps,
                        generator=generator,
                        controlnet_conditioning_image=img_tile,
                        height=img_tile.size[1],
                        width=img_tile.size[0],
                        controlnet_conditioning_scale=1.0,
                        alignment_ratio=refine_alignment_ratio,
                    ).images
                    results_tile+=x_samples_tile
    
        return results_tile, results, [full_segmask, mask_image], postive_prompt

    def download_image(url):
        response = requests.get(url)
        return Image.open(BytesIO(response.content)).convert("RGB")
