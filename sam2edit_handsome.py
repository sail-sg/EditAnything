# Edit Anything trained with Stable Diffusion + ControlNet + SAM  + BLIP2
import gradio as gr
from diffusers.utils import load_image
from sam2edit_lora import EditAnythingLoraModel, config_dict


def create_demo(process, process_image_click=None):

    examples = [
        ["1man, muscle,full body, vest, short straight hair, glasses, Gym, barbells, dumbbells, treadmills, boxing rings, squat racks, plates, dumbbell racks soft lighting, masterpiece, best quality, 8k uhd, film grain, Fujifilm XT3 photorealistic painting art by midjourney and greg rutkowski <lora:asianmale_v10:0.6>",
            "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck", 6],
        ["1man, 25 years- old, full body, wearing long-sleeve white shirt and tie, muscular rand black suit, soft lighting, masterpiece, best quality, 8k uhd, dslr, film grain, Fujifilm XT3 photorealistic painting art by midjourney and greg rutkowski <lora:asianmale_v10:0.6> <lora:uncutPenisLora_v10:0.6>",
            "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck", 6],
    ]

    print("The GUI is not fully tested yet. Please open an issue if you find bugs.")
    WARNING_INFO = f'''### [NOTE]  the model is collected from the Internet for demo only, please do not use it for commercial purposes.
    We are not responsible for possible risks using this model.
    Base model from https://huggingface.co/SG161222/Realistic_Vision_V2.0 Thanks!
    '''
    block = gr.Blocks()
    with block as demo:
        clicked_points = gr.State([])
        origin_image = gr.State(None)
        click_mask = gr.State(None)
        with gr.Row():
            gr.Markdown(
                "## Generate Your Handsome powered by EditAnything https://github.com/sail-sg/EditAnything ")
        with gr.Row():
            with gr.Column():

                with gr.Tab("Brush"):
                    source_image_brush = gr.Image(
                        source='upload',
                        label="Image (Upload an image and cover the region you want to edit with sketch)",
                        type="numpy", tool="sketch"
                    )
                    run_button = gr.Button(label="Run EditAnying")
                with gr.Tab("Click"):
                    source_image_click = gr.Image(
                        type="pil", interactive=True,
                        label="Image (Upload an image and click the region you want to edit)",
                    )

                    with gr.Row():
                        with gr.Row():
                            point_prompt = gr.Radio(
                                choices=["Foreground", "Background"],
                                value="Foreground",
                                label="Point Label",
                                interactive=True)
                        with gr.Row(scale=0.2):
                            clear_button_click = gr.Button(
                                value="Clear Click Points", interactive=True)
                            clear_button_image = gr.Button(
                                value="Clear Image", interactive=True)
                            run_button_click = gr.Button(
                                value="Run EditAnying", interactive=True)

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
        ips = [source_image_brush, enable_all_generate, mask_image, control_scale, enable_auto_prompt, prompt, a_prompt, n_prompt, num_samples, image_resolution,
               detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, enable_tile]
        run_button.click(fn=process, inputs=ips, outputs=[
            result_gallery, result_text])

        ip_click = [origin_image, enable_all_generate, click_mask, control_scale, enable_auto_prompt, prompt, a_prompt, n_prompt, num_samples, image_resolution,
                    detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, enable_tile]

        run_button_click.click(fn=process,
                               inputs=ip_click,
                               outputs=[result_gallery, result_text])

        source_image_click.upload(
            lambda image: image.copy() if image is not None else None,
            inputs=[source_image_click],
            outputs=[origin_image]
        )
        source_image_click.select(
            process_image_click,
            inputs=[origin_image, point_prompt,
                    clicked_points, image_resolution],
            outputs=[source_image_click, clicked_points, click_mask],
            show_progress=True, queue=True
        )
        clear_button_click.click(
            fn=lambda original_image: (original_image.copy(), [])
            if original_image is not None else (None, []),
            inputs=[origin_image],
            outputs=[source_image_click, clicked_points]
        )
        clear_button_image.click(
            fn=lambda: (None, [], None, None),
            inputs=[],
            outputs=[source_image_click, clicked_points,
                     result_gallery, result_text]
        )
        with gr.Row():
            ex = gr.Examples(examples=examples, fn=process,
                             inputs=[a_prompt, n_prompt, scale],
                             outputs=[result_gallery],
                             cache_examples=False)
        with gr.Row():
            gr.Markdown(WARNING_INFO)
    return demo


if __name__ == '__main__':
    model = EditAnythingLoraModel(base_model_path='../../gradio-rel/EditAnything/models/Realistic_Vision_V2.0',
                                  lora_model_path='../../gradio-rel/EditAnything/models/asianmale', use_blip=True)
    demo = create_demo(model.process, model.process_image_click)
    demo.queue().launch(server_name='0.0.0.0')
