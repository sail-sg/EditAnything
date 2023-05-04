import gradio as gr


from sam2edit import create_demo as create_demo_edit_anything
from sam2image import create_demo as create_demo_generate_anything
from sam2edit_beauty import create_demo as create_demo_beauty
from sam2edit_handsome import create_demo as create_demo_handsome
from sam2edit_lora_sd15_v3 import EditAnythingLoraModel


DESCRIPTION = '# [Edit Anything](https://github.com/sail-sg/EditAnything)'

with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.TabItem(' 🖌 Edit Anything'):
            create_demo_edit_anything()
        with gr.TabItem(' 👩‍🦰Beauty Edit/Generation'):
            model = EditAnythingLoraModel(base_model_path= '../../gradio-rel/EditAnything/models/chilloutmix_NiPrunedFp32Fix',lora_model_path= '../../gradio-rel/EditAnything/models/mix4', use_blip=True)
            create_demo_beauty(model.process)
        with gr.TabItem(' 👨‍🌾Handsome Edit/Generation'):
            model = EditAnythingLoraModel(base_model_path= '../../gradio-rel/EditAnything/models/Realistic_Vision_V2.0',
                        lora_model_path=None, use_blip=True)
            create_demo_handsome(model.process)
        # with gr.TabItem('Generate Anything'):
        #     create_demo_generate_anything()


demo.queue(api_open=False).launch()