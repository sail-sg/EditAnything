# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import os.path
import math
from PIL import Image

import numpy as np

import torch
import torch.utils.data as data

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from glip.maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from maskrcnn_benchmark.config import cfg


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= cfg.DATALOADER.MIN_KPS_PER_IMS:
        return True
    return False


def pil_loader(path, retry=5):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    ri = 0
    while ri < retry:
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except:
            ri += 1


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


class CocoDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index, return_meta=False):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        if isinstance(img_id, str):
            img_id = [img_id]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        meta = coco.loadImgs(img_id)[0]
        path = meta['file_name']
        img = pil_loader(os.path.join(self.root, path))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if return_meta:
            return img, target, meta
        else:
            return img, target

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class COCODataset(CocoDetection):
    def __init__(self, ann_file, root, remove_images_without_annotations, transforms=None, ignore_crowd=True,
                 max_box=-1,
                 few_shot=0, one_hot=False, override_category=None, **kwargs
                 ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                if isinstance(img_id, str):
                    ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
                else:
                    ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        if few_shot:
            ids = []
            cats_freq = [few_shot]*len(self.coco.cats.keys())
            if 'shuffle_seed' in kwargs and kwargs['shuffle_seed'] != 0:
                import random
                random.Random(kwargs['shuffle_seed']).shuffle(self.ids)
                print("Shuffle the dataset with random seed: ", kwargs['shuffle_seed'])
            for img_id in self.ids:
                if isinstance(img_id, str):
                    ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
                else:
                    ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                cat = set([ann['category_id'] for ann in anno]) #set/tuple corresponde to instance/image level
                is_needed = sum([cats_freq[c-1]>0 for c in cat])
                if is_needed:
                    ids.append(img_id)
                    for c in cat:
                        cats_freq[c-1] -= 1
                    # print(cat, cats_freq)
            self.ids = ids
        
        if override_category is not None:
            self.coco.dataset["categories"] = override_category
            print("Override category: ", override_category)

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms
        self.ignore_crowd = ignore_crowd
        self.max_box = max_box
        self.one_hot = one_hot

    def categories(self, no_background=True):
        categories = self.coco.dataset["categories"]
        label_list = {}
        for index, i in enumerate(categories):
            if not no_background or (i["name"] != "__background__" and i['id'] != 0):
                label_list[self.json_category_id_to_contiguous_id[i["id"]]] = i["name"]
        return label_list

    def __getitem__(self, idx):

        
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        if self.ignore_crowd:
            anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        if self.max_box > 0 and len(boxes) > self.max_box:
            rand_idx = torch.randperm(self.max_box)
            boxes = boxes[rand_idx, :]
        else:
            rand_idx = None
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)

        if rand_idx is not None:
            classes = classes[rand_idx]
        if cfg.DATASETS.CLASS_AGNOSTIC:
            classes = torch.ones_like(classes)
        target.add_field("labels", classes)

        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size, mode='poly')
            target.add_field("masks", masks)

        if anno and "cbox" in anno[0]:
            cboxes = [obj["cbox"] for obj in anno]
            cboxes = torch.as_tensor(cboxes).reshape(-1, 4)  # guard against no boxes
            cboxes = BoxList(cboxes, img.size, mode="xywh").convert("xyxy")
            target.add_field("cbox", cboxes)

        if anno and "keypoints" in anno[0]:
            keypoints = []
            gt_keypoint = self.coco.cats[1]['keypoints']  # <TODO> a better way to get keypoint description
            use_keypoint = cfg.MODEL.ROI_KEYPOINT_HEAD.KEYPOINT_NAME
            for obj in anno:
                if len(use_keypoint) > 0:
                    kps = []
                    for name in use_keypoint:
                        kp_idx = slice(3 * gt_keypoint.index(name), 3 * gt_keypoint.index(name) + 3)
                        kps += obj["keypoints"][kp_idx]
                    keypoints.append(kps)
                else:
                    keypoints.append(obj["keypoints"])
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if cfg.DATASETS.SAMPLE_RATIO != 0.0:
            ratio = cfg.DATASETS.SAMPLE_RATIO
            num_sample_target = math.ceil(len(target) * ratio) if ratio > 0 else math.ceil(-ratio)
            sample_idx = torch.randperm(len(target))[:num_sample_target]
            target = target[sample_idx]
        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
