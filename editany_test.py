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
            "dudou,1girl, beautiful face, solo, candle, brown hair, long hair, <lora:flowergirl:0.9>,ulzzang-6500-v1.1,(raw photo:1.2),((photorealistic:1.4))best quality ,masterpiece, illustration, an extremely delicate and beautiful, extremely detailed ,CG ,unity ,8k wallpaper, Amazing, finely detail, masterpiece,best quality,official art,extremely detailed CG unity 8k wallpaper,absurdres, incredibly absurdres, huge filesize, ultra-detailed, highres, extremely detailed,beautiful detailed girl, extremely detailed eyes and face, beautiful detailed eyes,cinematic lighting,1girl,see-through,looking at viewer,full body,full-body shot,outdoors,arms behind back,(chinese clothes) <lora:cuteGirlMix4_v10:1>",
            "(((mole))),sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, bad anatomy,(long hair:1.4),DeepNegative,(fat:1.2),facing away, looking away,tilted head, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit, extra arms, extra leg, extra foot,(freckles),(mole:2)",
            5,
        ],
        [
            "best quality, ultra high res, (photorealistic:1.4), (detailed beautiful girl:1.4), (medium breasts:0.8), looking_at_viewer, Detailed facial details, beautiful detailed eyes, (multicolored|blue|pink hair: 1.2), green eyes, slender, haunting smile, (makeup:0.3), red lips, <lora:cuteGirlMix4_v10:0.7>, highly detailed clothes, (ulzzang-6500-v1.1:0.3)",
            "EasyNegative, paintings, sketches, ugly, 3d, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, manboobs, backlight,(ugly:1.3), (duplicate:1.3), (morbid:1.2), (mutilated:1.2), (tranny:1.3), mutated hands, (poorly drawn hands:1.3), blurry, (bad anatomy:1.2), (bad proportions:1.3), extra limbs, (disfigured:1.3), (more than 2 nipples:1.3), (more than 1 navel:1.3), (missing arms:1.3), (extra legs:1.3), (fused fingers:1.6), (too many fingers:1.6), (unclear eyes:1.3), bad hands, missing fingers, extra digit, (futa:1.1), bad body, double navel, mutad arms, hused arms, (puffy nipples, dark areolae, dark nipples, rei no himo, inverted nipples, long nipples), NG_DeepNegative_V1_75t, pubic hair, fat rolls, obese, bad-picture-chill-75v",
            8,
        ],
        [
            "best quality, ultra high res, (photorealistic:1.4), (detailed beautiful girl:1.4), (medium breasts:0.8), looking_at_viewer, Detailed facial details, beautiful detailed eyes, (blue|pink hair), green eyes, slender, smile, (makeup:0.4), red lips, (full body, sitting, beach), <lora:cuteGirlMix4_v10:0.7>, highly detailed clothes, (ulzzang-6500-v1.1:0.3)",
            "asyNegative, paintings, sketches, ugly, 3d, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, manboobs, backlight,(ugly:1.3), (duplicate:1.3), (morbid:1.2), (mutilated:1.2), (tranny:1.3), mutated hands, (poorly drawn hands:1.3), blurry, (bad anatomy:1.2), (bad proportions:1.3), extra limbs, (disfigured:1.3), (more than 2 nipples:1.3), (more than 1 navel:1.3), (missing arms:1.3), (extra legs:1.3), (fused fingers:1.6), (too many fingers:1.6), (unclear eyes:1.3), bad hands, missing fingers, extra digit, (futa:1.1), bad body, double navel, mutad arms, hused arms, (puffy nipples, dark areolae, dark nipples, rei no himo, inverted nipples, long nipples), NG_DeepNegative_V1_75t, pubic hair, fat rolls, obese, bad-picture-chill-75v",
            7,
        ],
        [
            "mix4, whole body shot, ((8k, RAW photo, highest quality, masterpiece), High detail RAW color photo professional close-up photo, shy expression, cute, beautiful detailed girl, detailed fingers, extremely detailed eyes and face, beautiful detailed nose, beautiful detailed eyes, long eyelashes, light on face, looking at viewer, (closed mouth:1.2), 1girl, cute, young, mature face, (full body:1.3), ((small breasts)), realistic face, realistic body, beautiful detailed thigh,s, same eyes color, (realistic, photo realism:1. 37), (highest quality), (best shadow), (best illustration), ultra high resolution, physics-based rendering, cinematic lighting), solo, 1girl, highly detailed, in office, detailed office, open cardigan, ponytail contorted, beautiful eyes ,sitting in office,dating, business suit, cross-laced clothes, collared shirt, beautiful breast, small breast, Chinese dress, white pantyhose, natural breasts, pink and white hair, <lora:cuteGirlMix4_v10:1>",
            "paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), cloth, underwear, bra, low-res, normal quality, ((monochrome)), ((grayscale)), skin spots, acne, skin blemishes, age spots, glans, bad nipples, long nipples, bad vagina, extra fingers,fewer fingers,strange fingers,bad hand, ng_deepnegative_v1_75t, bad-picture-chill-75v",
            7,
        ],
    ]
    INFO = f"""
    ## Generate Your Beauty powered by EditAnything https://github.com/sail-sg/EditAnything
    This model is good at generating beautiful female.
    """
    WARNING_INFO = f"""### [NOTE]  the model is collected from the Internet for demo only, please do not use it for commercial purposes.
    We are not responsible for possible risks using this model.
    Lora model from https://civitai.com/models/14171/cutegirlmix4 Thanks!
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
    # sd_models_path = snapshot_download("shgao/sdmodels")
    # lora_model_path = hf_hub_download(
    #     "mlida/Cute_girl_mix4", "cuteGirlMix4_v10.safetensors")
    # model = EditAnythingLoraModel(base_model_path="andite/anything-v4.0",
    #                               lora_model_path=None, use_blip=True, extra_inpaint=True,
    #                               lora_weight=0.5,
    #                               )
    sd_models_path = snapshot_download("shgao/sdmodels")
    lora_model_path = hf_hub_download(
        "mlida/Cute_girl_mix4", "cuteGirlMix4_v10.safetensors"
    )
    model = EditAnythingLoraModel(
        base_model_path=os.path.join(
            sd_models_path, "chilloutmix_NiPrunedFp32Fix"),
        lora_model_path=lora_model_path,
        use_blip=True,
        extra_inpaint=True,
        lora_weight=0.5,
    )
    demo = create_demo(model.process, model.process_image_click)
    demo.queue().launch(server_name="0.0.0.0", share=True)
