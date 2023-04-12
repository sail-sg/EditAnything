import os
import os.path
import json
from PIL import Image

import torch
import torchvision
import torch.utils.data as data
from maskrcnn_benchmark.structures.bounding_box import BoxList

class Background(data.Dataset):
    """ Background

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
    """

    def __init__(self, ann_file, root, remove_images_without_annotations=None, transforms=None):
        self.root = root

        with open(ann_file, 'r') as f:
            self.ids = json.load(f)['images']
        self.transform = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        im_info = self.ids[index]
        path = im_info['file_name']
        fp = os.path.join(self.root, path)

        img = Image.open(fp).convert('RGB')
        if self.transform is not None:
            img, _ = self.transform(img, None)
        null_target = BoxList(torch.zeros((0,4)), (img.shape[-1], img.shape[-2]))
        null_target.add_field('labels', torch.zeros(0))

        return img, null_target, index

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        im_info = self.ids[index]
        return im_info