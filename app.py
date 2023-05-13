import gradio as gr
import os

from sam2edit import create_demo as create_demo_edit_anything
from sam2image import create_demo as create_demo_generate_anything
from sam2edit_beauty import create_demo as create_demo_beauty
from sam2edit_handsome import create_demo as create_demo_handsome
from sam2edit_lora import EditAnythingLoraModel, init_sam_model, init_blip_processor, init_blip_model
from huggingface_hub import hf_hub_download, snapshot_download

DESCRIPTION = f'''# [Edit Anything](https://github.com/sail-sg/EditAnything)
**Edit anything and keep the layout by segmenting anything in the image.**
'''

sam_generator, mask_predictor = init_sam_model()
blip_processor = init_blip_processor()
blip_model = init_blip_model()

sd_models_path = snapshot_download("shgao/sdmodels")

with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.TabItem('🖌Edit Anything'):
            model = EditAnythingLoraModel(base_model_path="stabilityai/stable-diffusion-2-inpainting",
                                          controlmodel_name='LAION Pretrained(v0-4)-SD21',
                                          lora_model_path=None, use_blip=True, extra_inpaint=False,
                                          sam_generator=sam_generator,
                                          mask_predictor=mask_predictor,
                                          blip_processor=blip_processor,
                                          blip_model=blip_model)
            create_demo_edit_anything(model.process, model.process_image_click)
        with gr.TabItem(' 👩‍🦰Beauty Edit/Generation'):
            lora_model_path = hf_hub_download(
                "mlida/Cute_girl_mix4", "cuteGirlMix4_v10.safetensors")
            model = EditAnythingLoraModel(base_model_path=os.path.join(sd_models_path, "chilloutmix_NiPrunedFp32Fix"),
                                          lora_model_path=lora_model_path, use_blip=True, extra_inpaint=True,
                                          sam_generator=sam_generator,
                                          mask_predictor=mask_predictor,
                                          blip_processor=blip_processor,
                                          blip_model=blip_model,
                                          lora_weight=0.5,
                                          )
            create_demo_beauty(model.process, model.process_image_click)
        with gr.TabItem(' 👨‍🌾Handsome Edit/Generation'):
            model = EditAnythingLoraModel(base_model_path=os.path.join(sd_models_path, "Realistic_Vision_V2.0"),
                                          lora_model_path=None, use_blip=True, extra_inpaint=True,
                                          sam_generator=sam_generator,
                                          mask_predictor=mask_predictor,
                                          blip_processor=blip_processor,
                                          blip_model=blip_model)
            create_demo_handsome(model.process, model.process_image_click)
        # with gr.TabItem('Generate Anything'):
        #     create_demo_generate_anything()
    # with gr.Tabs():
    #     gr.Markdown(SHARED_UI_WARNING)

demo.queue(api_open=False).launch(server_name='0.0.0.0', share=True)
