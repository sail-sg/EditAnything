# Edit Anything by Segment-Anything

[![HuggingFace space](https://img.shields.io/badge/🤗-HuggingFace%20Space-cyan.svg)](https://huggingface.co/spaces/shgao/EditAnything)

This is an ongoing project aims to **Edit and Generate Anything** in an image,
powered by [Segment Anything](https://github.com/facebookresearch/segment-anything), [ControlNet](https://github.com/lllyasviel/ControlNet),
[BLIP2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2), [Stable Diffusion](https://huggingface.co/spaces/stabilityai/stable-diffusion), etc.

Any forms of contribution and suggestion
are very welcomed!

# News🔥
2023/08/09 - Revise UI and code, fixed multiple known issues.

2023/07/25 - EditAnything is accepted by the ACM MM demo track.

2023/06/09 - Support cross-image region drag and merge, unleash creative fusion!

2023/05/24 - Support multiple high-quality character editing: clothes, haircut, colored contact lenses.

2023/05/22 - Support sketch to image by adjusting mask align strength in `sketch2image.py`!

2023/05/13 - Support interactive segmentation with click operation!  

2023/05/11 - Support tile model for detail refinement!

2023/05/04 - New demos of Beauty/Handsome Edit/Generation is released!

2023/05/04 - ControlNet-based inpainting model on any lora model is supported now. EditAnything can operate on any base/lord models without the requirements of inpainting model.

<details>
  <summary> More update logs. </summary>

    
2023/05/01 - Models V0.4 based on Stable Diffusion 1.5/2.1 are released. New models are trained with more data and iterations.[Model Zoo](https://github.com/sail-sg/EditAnything#model-zoo)

2023/04/20 - We support the Customized editing with DreamBooth.

2023/04/17 - We support the SAM mask to semantic segmentation mask.

2023/04/17 - We support different alignment degrees bettween edited parts and the SAM mask, check it out on [DEMO](https://huggingface.co/spaces/shgao/EditAnything)!

2023/04/15 - [Gradio demo on Huggingface](https://huggingface.co/spaces/shgao/EditAnything) is released!

2023/04/14 - New model trained with LAION dataset is released.

2023/04/13 - Support pretrained model auto downloading and gradio in `sam2image.py`.

2023/04/12 - An initial version of text-guided edit-anything is in `sam2groundingdino_edit.py`(object-level) and `sam2vlpart_edit.py`(part-level).

2023/04/10 - An initial version of edit-anything is in `sam2edit.py`.

2023/04/10 - We transfer the pretrained model into diffusers style, the pretrained model is auto loaded when using `sam2image_diffuser.py`. Now you can combine our pretrained model with different base models easily!

</details>

2023/04/09 - We released a pretrained model of StableDiffusion based ControlNet that generate images conditioned by SAM segmentation.

# Features

**Try our [![HuggingFace DEMO](https://img.shields.io/badge/🤗-HuggingFace%20Space-cyan.svg)](https://huggingface.co/spaces/shgao/EditAnything)🔥🔥🔥**
## Unleash creative fusion: Cross-image region drag and merge!🔥
<img width="1268" alt="image" src="https://github.com/sail-sg/EditAnything/assets/20515144/7f997db8-d2ea-4341-a7d7-dfe8ac5dd338">
<img width="1283" alt="image" src="https://github.com/sail-sg/EditAnything/assets/20515144/da34126c-d0fc-4020-85b6-1d99bed806e1">





## Clothes editing!🔥
<img width="1357" alt="image" src="https://github.com/sail-sg/EditAnything/assets/20515144/03452a0f-83ae-4257-995c-f3d8b71d4f1d">

## Haircut editing!🔥
<img width="1406" alt="image" src="https://github.com/sail-sg/EditAnything/assets/20515144/9091e3c9-c7e1-485d-bfe6-de5b21e83814">

## Colored contact lenses!🔥
<img width="1080" alt="image" src="https://github.com/sail-sg/EditAnything/assets/20515144/d9c8a136-e12c-4df4-aed0-a7c287e0ef3c">

## Human replacement with tile refinement!🔥

<img width="839" alt="image" src="https://github.com/sail-sg/EditAnything/assets/20515144/31883059-2bbc-442c-88aa-04d3a1da7abc">


## Draw your Sketch and Generate your Image!🔥
prompt: "a paint of  a  tree in the ground with a river."
<div>
<img width="250" alt="image" src="images/sk1.png">
<img width="250" alt="image" src="images/sk1_ex1.png">
<img width="250" alt="image" src="images/sk1_ex2.png">
</div>

<details>
  <summary> More demos. </summary>

prompt: "a paint, river, mountain, sun, cloud, beautiful field."
<div>
<img width="250" alt="image" src="images/sk4.png">
<img width="250" alt="image" src="images/sk4_ex1.png">
<img width="250" alt="image" src="images/sk4_ex2.png">
</div>

prompt: "a man, midsplit center parting hair, HD."
<div>
<img width="250" alt="image" src="images/sk2.png">
<img width="250" alt="image" src="images/sk2_ex1.png">
<img width="250" alt="image" src="images/sk2_ex2.png">
</div>

prompt: "a woman, long hair, detailed facial details, photorealistic, HD, beautiful face, solo, candle, brown hair, blue eye."
<div>
<img width="250" alt="image" src="images/sk3.png">
<img width="250" alt="image" src="images/sk3_ex1.png">
<img width="250" alt="image" src="images/sk3_ex2.png">
</div>
</details>

Also, you could use the generated image and sam model to refine your sketch definitely!

## Generate/Edit your beauty!!!🔥🔥🔥
**Edit Your beauty and Generate Your beauty**
<div>
<img width="277" alt="image" src="images/beauty_edit.gif">
<img width="300" alt="image" src="images/beauty_demo.gif">
</div>


## Customized editing with layout alignment control.
<img width="1392" alt="image" src="https://user-images.githubusercontent.com/20515144/233339751-2c9e4ec8-e884-4c0e-95de-42512eccee85.png">
EditAnything+DreamBooth: Train a customized DreamBooth Model with `tools/train_dreambooth_inpaint.py` and replace the base model in `sam2edit.py` with the trained model.

## Image Editing with layout alignment control.
<img width="1040" alt="image" src="https://user-images.githubusercontent.com/20515144/233106460-14eb0e5a-cbc1-457d-aad3-a56796f7bee1.png">

## Keep the layout and Generate your season!
<div>
    <img src="images/paint.jpg" height=256  alt="original paint">
    <img src="images/seg.png" height=256  alt="SAM">
</div>

Human Prompt: "A paint of spring/summer/autumn/winter field."
<div>
    <img src="images/spring.png" height=256 alt="spring">
    <img src="images/summer.png" height=256 alt="summer">
    <img src="images/autumn.png" height=256 alt="autumn">
    <img src="images/winter.png" height=256 alt="winter">
</div>

## Edit Specific Thing by Text-Grounding and Segment-Anything
### Editing by Text-guided Part Mask
Text Grounding: "dog head"

Human Prompt: "cute dog"
![p](images/sample_dog_head.jpg)

<details>
  <summary> More demos. </summary>
    
Text Grounding: "cat eye"

Human Prompt: "A cute small humanoid cat"
![p](images/sample_cat_eye.jpg)
    
</details>

### Editing by Text-guided Object Mask
Text Grounding: "bench"

Human Prompt: "bench" 
![p](images/sample_bench.jpg)

## Edit Anything by Segment-Anything

Human Prompt: "esplendent sunset sky, red brick wall"
![p](images/edit_sample2.jpg)


<details>
  <summary> More demos. </summary>
    
Human Prompt: "chairs by the lake, sunny day, spring"
![p](images/edit_sample1.jpg)
    
</details>


## Generate Anything by Segment-Anything

BLIP2 Prompt: "a large white and red ferry"
![p](images/sample1.jpg)
(1:input image; 2: segmentation mask; 3-8: generated images.)

<details>
  <summary> More demos. </summary>

BLIP2 Prompt: "a cloudy sky"
![p](images/sample2.jpg)

BLIP2 Prompt: "a black drone flying in the blue sky"
![p](images/sample3.jpg)

 </details>

1) The human prompt and BLIP2 generated prompt build the text instruction.
2) The SAM model segment the input image to generate segmentation mask without category.
3) The segmentation mask and text instruction guide the image generation.

## Generate semantic labels for each SAM mask.
![p](images/sample_semantic.jpg)
```
python sam2semantic.py

```

Highlight features:
- Pretrained ControlNet with SAM mask as condition enables the image generation with fine-grained control.
- category-unrelated SAM mask enables more forms of editing and generation.
- BLIP2 text generation enables text guidance-free control.

# Setup

**Create a environment**

```bash
    conda env create -f environment.yaml
    conda activate control
```

**Install BLIP2 and SAM**

Put these models in `models` folder.
```bash
# BLIP2 and SAM will be audo installed by running app.py
pip install git+https://github.com/huggingface/transformers.git

pip install git+https://github.com/facebookresearch/segment-anything.git

# For text-guided editing
pip install git+https://github.com/openai/CLIP.git

pip install git+https://github.com/facebookresearch/detectron2.git

pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

**Download pretrained model**
```bash

# Segment-anything ViT-H SAM model will be auto downloaded. 

# BLIP2 model will be auto downloaded.

# Part Grounding Swin-Base Model.
wget https://github.com/Cheems-Seminar/segment-anything-and-name-it/releases/download/v1.0/swinbase_part_0a0000.pth

# Grounding DINO Model.
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

# Get pretrained model from huggingface. 
# No need to download this! But please install safetensors for reading the ckpt.

```


**Run Demo**
```bash
python app.py
# or
python editany.py
# or
python sam2image.py
# or
python sam2vlpart_edit.py
# or
python sam2groundingdino_edit.py
```


# Model Zoo

| Model | Features | Download Path |
|-------|----------|---------------|
|SAM Pretrained(v0-1) | Good Nature Sense | [shgao/edit-anything-v0-1-1](https://huggingface.co/shgao/edit-anything-v0-1-1) |
|LAION Pretrained(v0-3) | Good Face        | [shgao/edit-anything-v0-3](https://huggingface.co/shgao/edit-anything-v0-3)
|LAION Pretrained(v0-4) | Support StableDiffusion 1.5/2.1, More training data and iterations, Good Face        | [shgao/edit-anything-v0-4-sd15](https://huggingface.co/shgao/edit-anything-v0-4-sd15) [shgao/edit-anything-v0-4-sd21](https://huggingface.co/shgao/edit-anything-v0-4-sd21)


# Training

1. Generate training dataset with `dataset_build.py`.
2. Transfer stable-diffusion model with `tool_add_control_sd21.py`.
2. Train model with `sam_train_sd21.py`.


# Acknowledgement
```
@InProceedings{gao2023editanything,
  author = {Gao, Shanghua and Lin, Zhijie and Xie, Xingyu and Zhou, Pan and Cheng, Ming-Ming and Yan, Shuicheng},
  title = {EditAnything: Empowering Unparalleled Flexibility in Image Editing and Generation},
  booktitle = {Proceedings of the 31st ACM International Conference on Multimedia, Demo track},
  year = {2023},
}
```

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
