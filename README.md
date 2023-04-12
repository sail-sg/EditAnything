# Edit Anything by Segment-Anything

This is an ongoing project aims to **Edit and Generate Anything** in an image,
powered by [Segment Anything](https://github.com/facebookresearch/segment-anything), [ControlNet](https://github.com/lllyasviel/ControlNet),
[BLIP2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2), [Stable Diffusion](https://huggingface.co/spaces/stabilityai/stable-diffusion), etc.

A project for fun. 
Any forms of contribution and suggestion
are very welcomed!



# News

2023/04/12 - An initial version of text-guided edit-anything is in `sam2groundingdino_edit.py`(object-level) and `sam2vlpart_edit.py`(part-level).

2023/04/10 - An initial version of edit-anything is in `sam2edit.py`.

2023/04/10 - We transfer the pretrained model into diffusers style, the pretrained model is auto loaded when using `sam2image_diffuser.py`. Now you can combine our pretrained model with different base models easily!

2023/04/09 - We released a pretrained model of StableDiffusion based ControlNet that generate images conditioned by SAM segmentation.

# Features

Highlight features:
- Pretrained ControlNet with SAM mask as condition enables the image generation with fine-grained control.
- category-unrelated SAM mask enables more forms of editing and generation.
- BLIP2 text generation enables text guidance-free control.

## Edit Specific Thing by Text-Grounding and Segment-Anything
### Part Grounding
Text Grounding: "dog head"

Human Prompt: "cute dog"
![p](images/sample_dog_head.jpg)

### Object Grounding
Text Grounding: "bench"

Human Prompt: "bench"
![p](images/sample_bench.jpg)

## Edit Anything by Segment-Anything

Human Prompt: "esplendent sunset sky, red brick wall"
![p](images/edit_sample2.jpg)

Human Prompt: "chairs by the lake, sunny day, spring"
![p](images/edit_sample1.jpg)
An initial version of edit-anything. (We will add more controls on masks very soon.)


## Generate Anything by Segment-Anything

BLIP2 Prompt: "a large white and red ferry"
![p](images/sample1.jpg)
(1:input image; 2: segmentation mask; 3-8: generated images.)

BLIP2 Prompt: "a cloudy sky"
![p](images/sample2.jpg)

BLIP2 Prompt: "a black drone flying in the blue sky"
![p](images/sample3.jpg)


1) The human prompt and BLIP2 generated prompt build the text instruction.
2) The SAM model segment the input image to generate segmentation mask without category.
3) The segmentation mask and text instruction guide the image generation.

Note: Due to the privacy protection in the SAM dataset,
faces in generated images are also blurred. We are training new models
with unblurred images to solve this.


# Ongoing

- [x] Conditional Generation trained with 85k samples in SAM dataset.

- [ ] Training with more images from LAION and SAM.

- [ ] Interactive control on different masks for image editing.

- [ ] Using [Grounding DINO](https://github.com/IDEA-Research/Grounded-Segment-Anything) for category-related auto editing. 

- [ ] ChatGPT guided image editing.



# Setup

**Create a environment**

```bash
    conda env create -f environment.yaml
    conda activate control
```

**Install BLIP2 and SAM**

Put these models in `models` folder.
```bash
pip install git+https://github.com/huggingface/transformers.git

pip install git+https://github.com/facebookresearch/segment-anything.git

# For text-guided editing
pip install git+https://github.com/openai/CLIP.git

pip install git+https://github.com/facebookresearch/detectron2.git

pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

**Download pretrained model**
```bash

# Segment-anything ViT-H SAM model. 
cd models/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# BLIP2 model will be auto downloaded.

# Part Grounding Swin-Base Model.
wget https://github.com/Cheems-Seminar/segment-anything-and-name-it/releases/download/v1.0/swinbase_part_0a0000.pth

# Grounding DINO Model.
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

# Get edit-anything-ckpt-v0-1.ckpt pretrained model from huggingface. 
# No need to download this if your are using sam2image_diffuser.py!!! But please install safetensors for reading the ckpt.
https://huggingface.co/shgao/edit-anything-v0-1

```


**Run Demo**
```bash
python sam2image_diffuser.py
# or 
python sam2image.py
# or 
python sam2edit.py
# or
python sam2vlpart_edit.py
# or
python sam2groundingdino_edit.py
```
Set 'use_gradio = True' in these files if you
have GUI to run the gradio demo.


# Training

1. Generate training dataset with `dataset_build.py`.
2. Transfer stable-diffusion model with `tool_add_control_sd21.py`.
2. Train model with `sam_train_sd21.py`.


# Acknowledgement
This project is based on:

[Segment Anything](https://github.com/facebookresearch/segment-anything),
[ControlNet](https://github.com/lllyasviel/ControlNet),
[BLIP2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2),
[MDT](https://github.com/sail-sg/MDT),
[Stable Diffusion](https://huggingface.co/spaces/stabilityai/stable-diffusion),
[Large-scale Unsupervised Semantic Segmentation](https://github.com/LUSSeg),
[Grounded Segment Anything: From Objects to Parts](https://github.com/Cheems-Seminar/segment-anything-and-name-it),
[Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)

Thanks for these amazing projects!
