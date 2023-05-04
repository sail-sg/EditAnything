# Edit Anything trained with Stable Diffusion + ControlNet + SAM  + BLIP2
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
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
from annotator.util import resize_image, HWC3


def create_demo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_blip = True
    use_gradio = True

    # Diffusion init using diffusers.
    # diffusers==0.14.0 required.

    # base_model_path = "stabilityai/stable-diffusion-2-1"
    base_model_path = "../dreambooth-concept"
    config_dict = OrderedDict([('SAM Pretrained(v0-1)', 'shgao/edit-anything-v0-1-1'),
                               ('LAION Pretrained(v0-3)', 'shgao/edit-anything-v0-3'),
                               ])

    def obtain_generation_model(controlnet_path):
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_path, controlnet=controlnet, torch_dtype=torch.float16
        )
        # speed up diffusion process with faster scheduler and memory optimization
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # remove following line if xformers is not installed
        pipe.enable_xformers_memory_efficient_attention()
        # pipe.enable_model_cpu_offload() # disable for now because of unknow bug in accelerate
        pipe.to(device)
        return pipe

    global default_controlnet_path
    default_controlnet_path = config_dict['LAION Pretrained(v0-3)']
    pipe = obtain_generation_model(default_controlnet_path)

    # Segment-Anything init.
    # pip install git+https://github.com/facebookresearch/segment-anything.git
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        print('segment_anything not installed')
        result = subprocess.run(['pip', 'install', 'git+https://github.com/facebookresearch/segment-anything.git'],
                                check=True)
        print(f'Install segment_anything {result}')
    if not os.path.exists('./models/sam_vit_h_4b8939.pth'):
        result = subprocess.run(
            ['wget', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth', '-P', 'models'],
            check=True)
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
        full_img = Image.fromarray(np.uint8(full_img))
        return full_img, res

    def get_sam_control(image):
        masks = mask_generator.generate(image)
        full_img, res = show_anns(masks)
        return full_img, res

    def process(condition_model, input_image, enable_auto_prompt, prompt, a_prompt, n_prompt, num_samples,
                image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):

        global default_controlnet_path
        global pipe
        print("To Use:", config_dict[condition_model], "Current:", default_controlnet_path)
        if default_controlnet_path != config_dict[condition_model]:
            print("Change condition model to:", config_dict[condition_model])
            pipe = obtain_generation_model(config_dict[condition_model])
            default_controlnet_path = config_dict[condition_model]

        with torch.no_grad():
            if use_blip and (enable_auto_prompt or len(prompt) == 0):
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

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)
            print("control.shape", control.shape)
            generator = torch.manual_seed(seed)
            x_samples = pipe(
                prompt=[prompt + ', ' + a_prompt] * num_samples,
                negative_prompt=[n_prompt] * num_samples,
                num_images_per_prompt=num_samples,
                num_inference_steps=ddim_steps,
                generator=generator,
                height=H,
                width=W,
                image=control.type(torch.float16),
            ).images

            results = [x_samples[i] for i in range(num_samples)]
        return [full_segmask] + results, prompt

    # disable gradio when not using GUI.
    if not use_gradio:
        # This part is not updated, it's just a example to use it without GUI.
        condition_model = 'shgao/edit-anything-v0-1-1'
        image_path = "images/sa_309398.jpg"
        input_image = Image.open(image_path)
        input_image = np.array(input_image, dtype=np.uint8)
        prompt = ""
        a_prompt = 'best quality, extremely detailed'
        n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
        num_samples = 4
        image_resolution = 512
        detect_resolution = 512
        ddim_steps = 100
        guess_mode = False
        strength = 1.0
        scale = 9.0
        seed = 10086
        eta = 0.0

        outputs, full_text = process(condition_model, input_image, prompt, a_prompt, n_prompt, num_samples,
                                     image_resolution,
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
        block = gr.Blocks()
        with block as demo:
            with gr.Row():
                gr.Markdown(
                    "## Generate Anything")
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(source='upload', type="numpy")
                    prompt = gr.Textbox(label="Prompt (Optional)")
                    run_button = gr.Button(label="Run")
                    condition_model = gr.Dropdown(choices=list(config_dict.keys()),
                                                  value=list(config_dict.keys())[0],
                                                  label='Model',
                                                  multiselect=False)
                    num_samples = gr.Slider(
                        label="Images", minimum=1, maximum=12, value=1, step=1)

                    enable_auto_prompt = gr.Checkbox(label='Auto generated BLIP2 prompt', value=True)
                    with gr.Accordion("Advanced options", open=False):
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
                        a_prompt = gr.Textbox(
                            label="Added Prompt", value='best quality, extremely detailed')
                        n_prompt = gr.Textbox(label="Negative Prompt",
                                              value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')

                with gr.Column():
                    result_gallery = gr.Gallery(
                        label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
                    result_text = gr.Text(label='BLIP2+Human Prompt Text')
            ips = [condition_model, input_image, enable_auto_prompt, prompt, a_prompt, n_prompt, num_samples,
                   image_resolution,
                   detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
            run_button.click(fn=process, inputs=ips, outputs=[result_gallery, result_text])
            return demo


if __name__ == '__main__':
    demo = create_demo()
    demo.queue().launch(server_name='0.0.0.0')
