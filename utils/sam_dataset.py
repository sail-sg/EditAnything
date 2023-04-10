import json
import cv2
import numpy as np
import os

from torch.utils.data import Dataset
import pycocotools.mask as maskUtils
from torchvision import transforms
import utils.transforms as custom_transforms
from PIL import Image
class SAMDataset(Dataset):
    def __init__(self, data_path='../data/files', txt_path='../data/data_85616.txt'):
        self.data = []
        with open(txt_path, 'rt') as f:
            for line in f:
                self.data.append(eval(line))
        self.data_path = data_path
        randomresizedcrop = custom_transforms.RandomResizedCrop(
            512,
            scale=(0.9, 1),
        )
        self.transform = custom_transforms.Compose([
            randomresizedcrop,
            custom_transforms.RandomHorizontalFlip(p=0.5),
            custom_transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])



    def __len__(self):
        return len(self.data)

    def load_rle_annotations_from_json(self, json_file_path, return_pil=True):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            anno_data = json.load(f)
        annotations = anno_data['annotations']
        height = int(anno_data['image']['height'])
        width = int(anno_data['image']['width'])

        map = np.zeros((height,width), dtype=np.uint16)
        for i in range(len(annotations)):
            ann = annotations[i]
            mask = maskUtils.decode(ann['segmentation'])
            map[mask != 0] = i + 1
        if return_pil:
            res = np.zeros((map.shape[0], map.shape[1], 3))
            res[:, :, 0] = map % 256
            res[:, :, 1] = map // 256
            res = Image.fromarray(res.astype(np.uint8))
            return res
        return map

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']


        source = self.load_rle_annotations_from_json(os.path.join(self.data_path, source_filename))
        target = Image.open(os.path.join(self.data_path, target_filename))


        target, source = self.transform(target, source)

        print(source.max(), source.min())
        target = target.permute(1,2,0)

        return dict(jpg=target, txt=prompt, hint=source)

