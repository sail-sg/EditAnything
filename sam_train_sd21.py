from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils.sam_dataset import SAMDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch


# Configs
resume_path = './models/control_sd21_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
data_path = '../data/files'
txt_path = '../data/data_85616.txt'


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = SAMDataset(data_path=data_path, txt_path=txt_path)
dataloader = DataLoader(dataset, num_workers=16,
                        batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=8, strategy="ddp", precision=32, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)
