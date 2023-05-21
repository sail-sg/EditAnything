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
import base64
from io import BytesIO

def create_demo():
    MAX_COLORS = 12
    canvas_html = "<div id='canvas-root' style='max-width:400px; margin: 0 auto'></div>"
    load_js = """
    async () => {
    const url = "https://huggingface.co/datasets/radames/gradio-components/raw/main/sketch-canvas.js"
    fetch(url)
      .then(res => res.text())
      .then(text => {
        const script = document.createElement('script');
        script.type = "module"
        script.src = URL.createObjectURL(new Blob([text], { type: 'application/javascript' }));
        document.head.appendChild(script);
      });
    }
    """

    get_js_colors = """
    async (canvasData) => {
      const canvasEl = document.getElementById("canvas-root");
      return [canvasEl._data]
    }
    """

    set_canvas_size = """
    async (aspect) => {
      if(aspect ==='square'){
        _updateCanvas(512,512)
      }
      if(aspect ==='horizontal'){
        _updateCanvas(768,512)
      }
      if(aspect ==='vertical'){
        _updateCanvas(512,768)
      }
    }
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # aspect = gr.Radio(["square", "horizontal", "vertical"], value="square", label="Aspect Ratio", visible=False if is_shared_ui else True)

    # Diffusion init using diffusers.
    # diffusers==0.14.0 required.

    base_model_path = "stabilityai/stable-diffusion-2-1"

    config_dict = OrderedDict([('SAM Pretrained(v0-1)', 'shgao/edit-anything-v0-1-1'),
                               ('LAION Pretrained(v0-3)', 'shgao/edit-anything-v0-3'),
                               ('LAION Pretrained(v0-4)', 'shgao/edit-anything-v0-4-sd21'),
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
    default_controlnet_path = config_dict['LAION Pretrained(v0-4)']
    pipe = obtain_generation_model(default_controlnet_path)

    def get_sam_control(image):
        return image

    from utils.sketch_helpers import get_high_freq_colors, color_quantization, create_binary_matrix

    def process_sketch(canvas_data):
        base64_img = canvas_data['image']
        image_data = base64.b64decode(base64_img.split(',')[1])
        image = Image.open(BytesIO(image_data)).convert("RGB")
        im2arr = np.array(image)
        colors = [tuple(map(int, rgb[4:-1].split(','))) for rgb in canvas_data['colors']]
        print(colors)
        colors_map, res = None, None
        ptr = 0
        for color in colors:
            r, g, b = color
            if any(c != 255 for c in (r, g, b)):
                binary_matrix = np.all(im2arr == (r, g, b), axis=-1)
                if colors_map is None:
                    colors_map = np.zeros((im2arr.shape[0], im2arr.shape[1]), dtype=np.uint16)
                    res = np.zeros((im2arr.shape[0], im2arr.shape[1], 3))
                colors_map[binary_matrix != 0] = ptr + 1
                ptr += 1
        res[:, :, 0] = colors_map % 256
        res[:, :, 1] = colors_map // 256
        res.astype(np.float32)
        # binary_matrixes['sketch'] = res
        return res, "sketch loaded."

    def process(condition_model, input_image, control_scale, prompt, a_prompt, n_prompt,
                num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):

        global default_controlnet_path
        global pipe
        print("To Use:", config_dict[condition_model], "Current:", default_controlnet_path)
        if default_controlnet_path != config_dict[condition_model]:
            print("Change condition model to:", config_dict[condition_model])
            pipe = obtain_generation_model(config_dict[condition_model])
            default_controlnet_path = config_dict[condition_model]

        with torch.no_grad():
            print("All text:", prompt)

            input_image = HWC3(input_image)

            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape

            # the default SAM model is trained with 1024 size.
            detected_map = get_sam_control(input_image)

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
                controlnet_conditioning_scale=float(control_scale),
                image=control.type(torch.float16),
            ).images

            results = [x_samples[i] for i in range(num_samples)]
        return results, prompt, "waiting for sketch..."

    # disable gradio when not using GUI.
    block = gr.Blocks()
    with block as demo:
        with gr.Row():
            gr.Markdown(
                "## Generate Anything")
        with gr.Row():
            with gr.Column():
                canvas_data = gr.JSON(value={}, visible=False)
                canvas = gr.HTML(canvas_html)
                aspect = gr.Radio(["square", "horizontal", "vertical"], value="square", label="Aspect Ratio",
                                  visible=False)
                button_run = gr.Button("I've finished my sketch", elem_id="main_button", interactive=True)
                result_text1 = gr.Text(label='sketch status:')

            with gr.Column(visible=True) as post_sketch:
                input_image = gr.Image(type="numpy", visible=False)
                prompt = gr.Textbox(label="Prompt (Optional)")
                run_button = gr.Button(label="Run")
                condition_model = gr.Dropdown(choices=list(config_dict.keys()),
                                              value=list(config_dict.keys())[0],
                                              label='Model',
                                              multiselect=False)
                control_scale = gr.Slider(
                    label="Mask Align strength", info="Large value -> strict alignment with SAM mask", minimum=0,
                    maximum=1, value=1, step=0.1)
                num_samples = gr.Slider(
                    label="Images", minimum=1, maximum=12, value=1, step=1)

                # enable_auto_prompt = True
                with gr.Accordion("Advanced options", open=False):
                    image_resolution = gr.Slider(
                        label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                    strength = gr.Slider(
                        label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                    guess_mode = gr.Checkbox(label='Guess Mode', value=False)
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
        aspect.change(None, inputs=[aspect], outputs=None, _js=set_canvas_size)
        button_run.click(process_sketch, inputs=[canvas_data],
                         outputs=[input_image, result_text1], _js=get_js_colors, queue=False)
        ips = [condition_model, input_image, control_scale, prompt, a_prompt, n_prompt,
               num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
        run_button.click(fn=process, inputs=ips, outputs=[result_gallery, result_text, result_text1])
        demo.load(None, None, None, _js=load_js)
        return demo


if __name__ == '__main__':
    demo = create_demo()
    demo.queue().launch(server_name='0.0.0.0', share=True)
