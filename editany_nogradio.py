import pickle
from editany_lora import EditAnythingLoraModel
model = EditAnythingLoraModel(
    base_model_path="runwayml/stable-diffusion-v1-5",
    controlmodel_name='LAION Pretrained(v0-4)-SD15',
    lora_model_path=None, use_blip=True, extra_inpaint=True,
)

with open('input_data.pkl', 'rb') as f:
    input_data = pickle.load(f)

print(input_data)
    
res = model.process(*input_data['args'], **input_data['kwargs'])

# a woman in a tan suit and white shirt

# best quality, extremely detailed,iron man wallpaper