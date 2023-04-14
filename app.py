import gradio as gr


from sam2edit import create_demo as create_demo_edit_anything
from sam2image import create_demo as create_demo_generate_anything


DESCRIPTION = '# [Edit Anything](https://github.com/sail-sg/EditAnything)'

with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.TabItem('Edit Anything'):
            create_demo_edit_anything()
        with gr.TabItem('Generate Anything'):
            create_demo_generate_anything()


demo.queue(api_open=False).launch()