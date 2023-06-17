# Edit Anything trained with Stable Diffusion + ControlNet + SAM  + BLIP2
import os
import gradio as gr
from diffusers.utils import load_image
from editany_lora import EditAnythingLoraModel, config_dict
from editany_demo import create_demo_template
from huggingface_hub import hf_hub_download, snapshot_download


def create_demo(process, process_image_click=None):

    examples = [
        [
            "1man, muscle,full body, vest, short straight hair, glasses, Gym, barbells, dumbbells, treadmills, boxing rings, squat racks, plates, dumbbell racks soft lighting, masterpiece, best quality, 8k uhd, film grain, Fujifilm XT3 photorealistic painting art by midjourney and greg rutkowski <lora:asianmale_v10:0.6>",
            "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
            6,
        ],
        [
            "1man, 25 years- old, full body, wearing long-sleeve white shirt and tie, muscular rand black suit, soft lighting, masterpiece, best quality, 8k uhd, dslr, film grain, Fujifilm XT3 photorealistic painting art by midjourney and greg rutkowski <lora:asianmale_v10:0.6> <lora:uncutPenisLora_v10:0.6>",
            "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
            6,
        ],
    ]

    print("The GUI is not fully tested yet. Please open an issue if you find bugs.")

    INFO = f"""
    ## Generate Your Handsome powered by EditAnything https://github.com/sail-sg/EditAnything 
    This model is good at generating handsome male.
    """
    WARNING_INFO = f"""### [NOTE]  the model is collected from the Internet for demo only, please do not use it for commercial purposes.
    We are not responsible for possible risks using this model.
    Base model from https://huggingface.co/SG161222/Realistic_Vision_V2.0 Thanks!
    """
    demo = create_demo_template(
        process,
        process_image_click,
        examples=examples,
        INFO=INFO,
        WARNING_INFO=WARNING_INFO,
    )
    return demo


if __name__ == "__main__":
    model = EditAnythingLoraModel(
        base_model_path="Realistic_Vision_V2.0", lora_model_path=None, use_blip=True
    )
    demo = create_demo(model.process, model.process_image_click)
    demo.queue().launch(server_name="0.0.0.0")
