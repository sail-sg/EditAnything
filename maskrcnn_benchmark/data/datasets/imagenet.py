import os
import os.path
import json
from PIL import Image

import torch.utils.data as data

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImageNet(data.Dataset):
    """ ImageNet

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
    """

    def __init__(self, ann_file, root, remove_images_without_annotations=None, transforms=None):


        self.root = root
        self.transform = transforms

        meta_file = os.path.join(root, ann_file)
        assert os.path.exists(meta_file), 'meta file %s under root %s not found' % (os.path.basename(meta_file), root)

        with open(meta_file, 'r') as f:
            meta = json.load(f)

        self.classes = meta['classes']
        self.class_to_idx = meta['class_to_idx']
        self.samples = meta['samples']
        self.num_sample = len(self.samples)
        self.allsamples = self.samples

    def select_class(self, cls):
        new_samples = [sample for sample in self.allsamples if sample[-1] in cls]
        self.samples = new_samples
        self.num_sample = len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        img_path, target = self.samples[index]
        sample = pil_loader(self.root + '/' + img_path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, index

    def __len__(self):
        return len(self.samples)