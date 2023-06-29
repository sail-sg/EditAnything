export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./tmp/textinv/img"
export OUTPUT_DIR="./tmp/textinv/model"

CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 1111 texutal_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<new-obj>" --initializer_token="mark" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --num_vectors 10