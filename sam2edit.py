# Edit Anything trained with Stable Diffusion + ControlNet + SAM  + BLIP2
import gradio as gr
from diffusers.utils import load_image
from sam2edit_lora import EditAnythingLoraModel, config_dict


def create_demo(process):



    print("The GUI is not fully tested yet. Please open an issue if you find bugs.")
    WARNING_INFO = f'''### [NOTE]  the model is collected from the Internet for demo only, please do not use it for commercial purposes.
    We are not responsible for possible risks using this model.
    '''
    block = gr.Blocks()
    with block as demo:
        with gr.Row():
            gr.Markdown(
                "## EditAnything https://github.com/sail-sg/EditAnything ")
        with gr.Row():
            with gr.Column():
                source_image = gr.Image(
                    source='upload', label="Image (Upload an image and cover the region you want to edit with sketch)",  type="numpy", tool="sketch")
                enable_all_generate = gr.Checkbox(
                    label='Auto generation on all region.', value=False)
                prompt = gr.Textbox(
                    label="Prompt (Text in the expected things of edited region)")
                enable_auto_prompt = gr.Checkbox(
                    label='Auto generate text prompt from input image with BLIP2: Warning: Enable this may makes your prompt not working.', value=False)
                a_prompt = gr.Textbox(
                    label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
                control_scale = gr.Slider(
                    label="Mask Align strength (Large value means more strict alignment with SAM mask)", minimum=0, maximum=1, value=1, step=0.1)
                run_button = gr.Button(label="Run")
                num_samples = gr.Slider(
                    label="Images", minimum=1, maximum=12, value=2, step=1)
                seed = gr.Slider(label="Seed", minimum=-1,
                                 maximum=2147483647, step=1, randomize=True)
                enable_tile = gr.Checkbox(
                    label='Tile refinement for high resolution generation.', value=True)
                with gr.Accordion("Advanced options", open=False):
                    mask_image = gr.Image(
                        source='upload', label="(Optional) Upload a predefined mask of edit region if you do not want to write your prompt.", type="numpy", value=None)
                    image_resolution = gr.Slider(
                        label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                    strength = gr.Slider(
                        label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                    guess_mode = gr.Checkbox(
                        label='Guess Mode', value=False)
                    detect_resolution = gr.Slider(
                        label="SAM Resolution", minimum=128, maximum=2048, value=1024, step=1)
                    ddim_steps = gr.Slider(
                        label="Steps", minimum=1, maximum=100, value=30, step=1)
                    scale = gr.Slider(
                        label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                    eta = gr.Number(label="eta (DDIM)", value=0.0)
            with gr.Column():
                result_gallery = gr.Gallery(
                    label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
                result_text = gr.Text(label='BLIP2+Human Prompt Text')
        ips = [source_image, enable_all_generate, mask_image, control_scale, enable_auto_prompt, prompt, a_prompt, n_prompt, num_samples, image_resolution,
               detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, enable_tile]
        run_button.click(fn=process, inputs=ips, outputs=[
            result_gallery, result_text])
        # with gr.Row():
        #     ex = gr.Examples(examples=examples, fn=process,
        #                      inputs=[a_prompt, n_prompt, scale],
        #                      outputs=[result_gallery],
        #                      cache_examples=False)
        with gr.Row():
            gr.Markdown(WARNING_INFO)
    return demo


if __name__ == '__main__':
    model = EditAnythingLoraModel(base_model_path="stabilityai/stable-diffusion-2-inpainting",
                                  controlmodel_name='LAION Pretrained(v0-4)-SD21', extra_inpaint=False,
                                  lora_model_path=None, use_blip=True)
    demo = create_demo(model.process)
    demo.queue().launch(server_name='0.0.0.0')
