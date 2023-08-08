# Edit Anything trained with Stable Diffusion + ControlNet + SAM  + BLIP2
import gradio as gr

import numpy as np
import cv2
from cv2 import imencode
import base64

def create_demo_template(
    process,
    process_image_click=None,
    examples=None,
    INFO="EditAnything https://github.com/sail-sg/EditAnything",
    WARNING_INFO=None,
    enable_auto_prompt_default=False,
):

    print("The GUI is not fully tested yet. Please open an issue if you find bugs.")
    block = gr.Blocks()
    with block as demo:
        clicked_points = gr.State([])
        origin_image = gr.State(None)
        click_mask = gr.State(None)
        ref_clicked_points = gr.State([])
        ref_origin_image = gr.State(None)
        ref_click_mask = gr.State(None)
        with gr.Row():
            gr.Markdown(INFO)
        with gr.Row(equal_height=False):
            with gr.Column():
                with gr.Tab("Click🖱"):
                    source_image_click = gr.Image(
                        type="pil",
                        interactive=True,
                        label="Image: Upload an image and click the region you want to edit.",
                    )
                    with gr.Column():
                        with gr.Row():
                            point_prompt = gr.Radio(
                                choices=["Foreground Point",
                                         "Background Point"],
                                value="Foreground Point",
                                label="Point Label",
                                interactive=True,
                                show_label=False,
                            )
                            with gr.Row():
                                clear_button_click = gr.Button(
                                    value="Clear Points", interactive=True
                                )
                                clear_button_image = gr.Button(
                                    value="Reset Image", interactive=True
                                )
                        with gr.Row():
                            run_button_click = gr.Button(
                                label="Run EditAnying", interactive=True
                            )
                with gr.Tab("Brush🖌️"):
                    source_image_brush = gr.Image(
                        source="upload",
                        label="Image: Upload an image and cover the region you want to edit with sketch",
                        type="numpy",
                        tool="sketch",
                        brush_color="#00FFBF"
                    )
                    run_button = gr.Button(
                        label="Run EditAnying", interactive=True)
                with gr.Tab("All region"):
                    source_image_clean = gr.Image(
                        source="upload",
                        label="Image: Upload an image",
                        type="numpy",
                    )
                    run_button_allregion = gr.Button(
                        label="Run EditAnying", interactive=True)
                with gr.Row():
                    # enable_all_generate = gr.Checkbox(
                    #     label="All Region Generation", value=False
                    # )
                    control_scale = gr.Slider(
                        label="SAM Mask Alignment Strength",
                        # info="Large value -> strict alignment with SAM mask",
                        minimum=0,
                        maximum=1,
                        value=0.5,
                        step=0.1,
                    )
                    with gr.Row():
                        num_samples = gr.Slider(
                            label="Images", minimum=1, maximum=12, value=2, step=1
                        )
                        seed = gr.Slider(
                            label="Seed",
                            minimum=-1,
                            maximum=2147483647,
                            step=1,
                            randomize=True,
                        )
                with gr.Column():
                    with gr.Row():
                        enable_auto_prompt = gr.Checkbox(
                            label="Prompt Auto Generation (Enable this may makes your prompt not working)",
                            # info="",
                            value=enable_auto_prompt_default,
                        )
                    with gr.Row():
                        a_prompt = gr.Textbox(
                            label="Positive Prompt",
                            info="Text in the expected things of edited region",
                            value="best quality, extremely detailed,",
                        )
                        n_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, NSFW",
                        )

                with gr.Row():
                    enable_tile = gr.Checkbox(
                        label="High-resolution Refinement",
                        info="Slow inference",
                        value=True,
                    )
                    refine_alignment_ratio = gr.Slider(
                        label="Similarity with Initial Results",
                        # info="Large value -> strict alignment with input image. Small value -> strong global consistency",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                    )

                with gr.Accordion("Cross-image Drag Options", open=False):
                    # ref_image = gr.Image(
                    #     source='upload', label="Upload a reference image", type="pil", value=None)
                    ref_image = gr.Image(
                        source="upload",
                        label="Upload a reference image and cover the region you want to use with sketch",
                        type="pil",
                        tool="sketch",
                        brush_color="#00FFBF",
                    )
                    with gr.Row():
                        ref_auto_prompt = gr.Checkbox(
                            label="Ref. Auto Prompt", value=True
                        )
                        ref_prompt = gr.Textbox(
                            label="Prompt",
                            info="Text in the prompt of edited region",
                            value="best quality, extremely detailed, ",
                        )
                    # ref_image = gr.Image(
                    #     type="pil", interactive=True,
                    #     label="Image: Upload an image and click the region you want to use as reference.",
                    # )
                    # with gr.Column():
                    #     with gr.Row():
                    #         ref_point_prompt = gr.Radio(
                    #             choices=["Foreground Point", "Background Point"],
                    #             value="Foreground Point",
                    #             label="Point Label",
                    #             interactive=True, show_label=False)
                    #         ref_clear_button_click = gr.Button(
                    #             value="Clear Click Points", interactive=True)
                    #         ref_clear_button_image = gr.Button(
                    #             value="Clear Image", interactive=True)
                    with gr.Row():
                        reference_attn = gr.Checkbox(
                            label="reference_attn", value=True)
                        reference_adain = gr.Checkbox(
                            label="reference_adain", value=True
                        )
                    with gr.Row():
                        ref_sam_scale = gr.Slider(
                            label="Pos Control Scale",
                            minimum=0,
                            maximum=1.0,
                            value=0.3,
                            step=0.1,
                        )
                        ref_inpaint_scale = gr.Slider(
                            label="Content Control Scale",
                            minimum=0,
                            maximum=1.0,
                            value=0.2,
                            step=0.1,
                        )

                    with gr.Row():
                        ref_textinv = gr.Checkbox(
                            label="Use textual inversion token", value=False
                        )
                        ref_textinv_path = gr.Textbox(
                            label="textual inversion token path",
                            info="Text in the inversion token path",
                            value=None,
                        )
                    with gr.Accordion("Advanced options", open=False):
                        style_fidelity = gr.Slider(
                            label="Style fidelity",
                            minimum=0,
                            maximum=1.,
                            value=0.,
                            step=0.1,
                        )
                        attention_auto_machine_weight = gr.Slider(
                            label="Attention Reference Weight",
                            minimum=0,
                            maximum=1.0,
                            value=1.0,
                            step=0.01,
                        )
                        gn_auto_machine_weight = gr.Slider(
                            label="GroupNorm Reference Weight",
                            minimum=0,
                            maximum=1.0,
                            value=1.0,
                            step=0.01,
                        )
                        ref_scale = gr.Slider(
                            label="Frequency Reference Guidance Scale",
                            minimum=0,
                            maximum=1.0,
                            value=0.0,
                            step=0.1,
                        )

                with gr.Accordion("Advanced Options", open=False):
                    mask_image = gr.Image(
                        source="upload",
                        label="Upload a predefined mask of edit region: Switch to Brush mode when using this!",
                        type="numpy",
                        value=None,
                    )
                    image_resolution = gr.Slider(
                        label="Image Resolution",
                        minimum=256,
                        maximum=768,
                        value=512,
                        step=64,
                    )
                    refine_image_resolution = gr.Slider(
                        label="Image Resolution",
                        minimum=256,
                        maximum=8192,
                        value=1024,
                        step=64,
                    )
                    guess_mode = gr.Checkbox(label="Guess Mode", value=False)
                    detect_resolution = gr.Slider(
                        label="SAM Resolution",
                        minimum=128,
                        maximum=2048,
                        value=1024,
                        step=1,
                    )
                    ddim_steps = gr.Slider(
                        label="Steps", minimum=1, maximum=100, value=30, step=1
                    )
                    scale = gr.Slider(
                        label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                    alpha_weight = gr.Slider(
                        label="Alpha weight", info="Alpha mixing with original image", minimum=0,
                        maximum=1, value=0.0, step=0.1)
                    use_scale_map = gr.Checkbox(
                        label='Use scale map', value=False)
                    eta = gr.Number(label="eta (DDIM)", value=0.0)
                    condition_model = gr.Textbox(
                        label="Condition model path",
                        info="Text in the Controlnet model path in hugglingface",
                        value="EditAnything",
                    )
            with gr.Column():
                result_gallery_refine = gr.Gallery(
                    label="Output High quality", show_label=True, elem_id="gallery", preview=False)
                result_gallery_init = gr.Gallery(
                    label="Output Low quality", show_label=True, elem_id="gallery", height="auto")
                result_gallery_ref = gr.Gallery(
                    label="Output Ref", show_label=False, elem_id="gallery", height="auto")
                result_text = gr.Text(label="ALL Prompt Text")

        ips = [
            source_image_brush,
            gr.State(False),  # enable_all_generate
            mask_image,
            control_scale,
            enable_auto_prompt,
            a_prompt,
            n_prompt,
            num_samples,
            image_resolution,
            detect_resolution,
            ddim_steps,
            guess_mode,
            scale,
            seed,
            eta,
            enable_tile,
            refine_alignment_ratio,
            refine_image_resolution,
            alpha_weight,
            use_scale_map,
            condition_model,
            ref_image,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            ref_prompt,
            ref_sam_scale,
            ref_inpaint_scale,
            ref_auto_prompt,
            ref_textinv,
            ref_textinv_path,
            ref_scale,
        ]
        run_button.click(
            fn=process,
            inputs=ips,
            outputs=[
                result_gallery_refine,
                result_gallery_init,
                result_gallery_ref,
                result_text,
            ],
        )
        ips_allregion = [
            source_image_clean,
            gr.State(True),  # enable_all_generate
            mask_image,
            control_scale,
            enable_auto_prompt,
            a_prompt,
            n_prompt,
            num_samples,
            image_resolution,
            detect_resolution,
            ddim_steps,
            guess_mode,
            scale,
            seed,
            eta,
            enable_tile,
            refine_alignment_ratio,
            refine_image_resolution,
            alpha_weight,
            use_scale_map,
            condition_model,
            ref_image,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            ref_prompt,
            ref_sam_scale,
            ref_inpaint_scale,
            ref_auto_prompt,
            ref_textinv,
            ref_textinv_path,
            ref_scale,
        ]
        run_button_allregion.click(
            fn=process,
            inputs=ips_allregion,
            outputs=[
                result_gallery_refine,
                result_gallery_init,
                result_gallery_ref,
                result_text,
            ],
        )

        ip_click = [
            origin_image,
            gr.State(False),  # enable_all_generate
            click_mask,
            control_scale,
            enable_auto_prompt,
            a_prompt,
            n_prompt,
            num_samples,
            image_resolution,
            detect_resolution,
            ddim_steps,
            guess_mode,
            scale,
            seed,
            eta,
            enable_tile,
            refine_alignment_ratio,
            refine_image_resolution,
            alpha_weight,
            use_scale_map,
            condition_model,
            ref_image,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            ref_prompt,
            ref_sam_scale,
            ref_inpaint_scale,
            ref_auto_prompt,
            ref_textinv,
            ref_textinv_path,
            ref_scale,
        ]

        run_button_click.click(
            fn=process,
            inputs=ip_click,
            outputs=[
                result_gallery_refine,
                result_gallery_init,
                result_gallery_ref,
                result_text,
            ],
        )

        source_image_click.upload(
            lambda image: image.copy() if image is not None else None,
            inputs=[source_image_click],
            outputs=[origin_image],
        )
        source_image_click.select(
            process_image_click,
            inputs=[origin_image, point_prompt,
                    clicked_points, image_resolution],
            outputs=[source_image_click, clicked_points, click_mask],
            show_progress=True,
            queue=True,
        )
        clear_button_click.click(
            fn=lambda original_image: (original_image.copy(), [], None)
            if original_image is not None
            else (None, [], None),
            inputs=[origin_image],
            outputs=[source_image_click, clicked_points, click_mask],
        )
        clear_button_image.click(
            fn=lambda: (None, [], None, None, None),
            inputs=[],
            outputs=[
                source_image_click,
                clicked_points,
                click_mask,
                result_gallery_init,
                result_text,
            ],
        )

        if examples is not None:
            with gr.Row():
                ex = gr.Examples(
                    examples=examples,
                    fn=process,
                    inputs=[a_prompt, n_prompt, scale],
                    outputs=[result_gallery_init],
                    cache_examples=False,
                )
        if WARNING_INFO is not None:
            with gr.Row():
                gr.Markdown(WARNING_INFO)
    return demo
