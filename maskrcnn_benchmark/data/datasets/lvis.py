# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import os
import time
from collections import defaultdict

import pycocotools.mask as mask_utils
import torchvision
from PIL import Image

# from .coco import ConvertCocoPolysToMask, make_coco_transforms
from .modulated_coco import ConvertCocoPolysToMask


def _isArrayLike(obj):
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


class LVIS:
    def __init__(self, annotation_path=None):
        """Class for reading and visualizing annotations.
        Args:
            annotation_path (str): location of annotation file
        """
        self.anns = {}
        self.cats = {}
        self.imgs = {}
        self.img_ann_map = defaultdict(list)
        self.cat_img_map = defaultdict(list)
        self.dataset = {}

        if annotation_path is not None:
            print("Loading annotations.")

            tic = time.time()
            self.dataset = self._load_json(annotation_path)
            print("Done (t={:0.2f}s)".format(time.time() - tic))

            assert type(self.dataset) == dict, "Annotation file format {} not supported.".format(type(self.dataset))
            self._create_index()

    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _create_index(self):
        print("Creating index.")

        self.img_ann_map = defaultdict(list)
        self.cat_img_map = defaultdict(list)

        self.anns = {}
        self.cats = {}
        self.imgs = {}

        for ann in self.dataset["annotations"]:
            self.img_ann_map[ann["image_id"]].append(ann)
            self.anns[ann["id"]] = ann

        for img in self.dataset["images"]:
            self.imgs[img["id"]] = img

        for cat in self.dataset["categories"]:
            self.cats[cat["id"]] = cat

        for ann in self.dataset["annotations"]:
            self.cat_img_map[ann["category_id"]].append(ann["image_id"])

        print("Index created.")

    def get_ann_ids(self, img_ids=None, cat_ids=None, area_rng=None):
        """Get ann ids that satisfy given filter conditions.
        Args:
            img_ids (int array): get anns for given imgs
            cat_ids (int array): get anns for given cats
            area_rng (float array): get anns for a given area range. e.g [0, inf]
        Returns:
            ids (int array): integer array of ann ids
        """
        if img_ids is not None:
            img_ids = img_ids if _isArrayLike(img_ids) else [img_ids]
        if cat_ids is not None:
            cat_ids = cat_ids if _isArrayLike(cat_ids) else [cat_ids]
        anns = []
        if img_ids is not None:
            for img_id in img_ids:
                anns.extend(self.img_ann_map[img_id])
        else:
            anns = self.dataset["annotations"]

        # return early if no more filtering required
        if cat_ids is None and area_rng is None:
            return [_ann["id"] for _ann in anns]

        cat_ids = set(cat_ids)

        if area_rng is None:
            area_rng = [0, float("inf")]

        ann_ids = [
            _ann["id"]
            for _ann in anns
            if _ann["category_id"] in cat_ids and _ann["area"] > area_rng[0] and _ann["area"] < area_rng[1]
        ]
        return ann_ids

    def get_cat_ids(self):
        """Get all category ids.
        Returns:
            ids (int array): integer array of category ids
        """
        return list(self.cats.keys())

    def get_img_ids(self):
        """Get all img ids.
        Returns:
            ids (int array): integer array of image ids
        """
        return list(self.imgs.keys())

    def _load_helper(self, _dict, ids):
        if ids is None:
            return list(_dict.values())
        elif _isArrayLike(ids):
            return [_dict[id] for id in ids]
        else:
            return [_dict[ids]]

    def load_anns(self, ids=None):
        """Load anns with the specified ids. If ids=None load all anns.
        Args:
            ids (int array): integer array of annotation ids
        Returns:
            anns (dict array) : loaded annotation objects
        """
        return self._load_helper(self.anns, ids)

    def load_cats(self, ids):
        """Load categories with the specified ids. If ids=None load all
        categories.
        Args:
            ids (int array): integer array of category ids
        Returns:
            cats (dict array) : loaded category dicts
        """
        return self._load_helper(self.cats, ids)

    def load_imgs(self, ids):
        """Load categories with the specified ids. If ids=None load all images.
        Args:
            ids (int array): integer array of image ids
        Returns:
            imgs (dict array) : loaded image dicts
        """
        return self._load_helper(self.imgs, ids)

    def download(self, save_dir, img_ids=None):
        """Download images from mscoco.org server.
        Args:
            save_dir (str): dir to save downloaded images
            img_ids (int array): img ids of images to download
        """
        imgs = self.load_imgs(img_ids)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for img in imgs:
            file_name = os.path.join(save_dir, img["file_name"])
            if not os.path.exists(file_name):
                from urllib.request import urlretrieve

                urlretrieve(img["coco_url"], file_name)

    def ann_to_rle(self, ann):
        """Convert annotation which can be polygons, uncompressed RLE to RLE.
        Args:
            ann (dict) : annotation object
        Returns:
            ann (rle)
        """
        img_data = self.imgs[ann["image_id"]]
        h, w = img_data["height"], img_data["width"]
        segm = ann["segmentation"]
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_utils.frPyObjects(segm, h, w)
            rle = mask_utils.merge(rles)
        elif isinstance(segm["counts"], list):
            # uncompressed RLE
            rle = mask_utils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann["segmentation"]
        return rle

    def ann_to_mask(self, ann):
        """Convert annotation which can be polygons, uncompressed RLE, or RLE
        to binary mask.
        Args:
            ann (dict) : annotation object
        Returns:
            binary mask (numpy 2D array)
        """
        rle = self.ann_to_rle(ann)
        return mask_utils.decode(rle)


class LvisDetectionBase(torchvision.datasets.VisionDataset):
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(LvisDetectionBase, self).__init__(root, transforms, transform, target_transform)
        self.lvis = LVIS(annFile)
        self.ids = list(sorted(self.lvis.imgs.keys()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        lvis = self.lvis
        img_id = self.ids[index]
        ann_ids = lvis.get_ann_ids(img_ids=img_id)
        target = lvis.load_anns(ann_ids)

        path = "/".join(self.lvis.load_imgs(img_id)[0]["coco_url"].split("/")[-2:])

        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    

    def __len__(self):
        return len(self.ids)


class LvisDetection(LvisDetectionBase):
    def __init__(self, img_folder, ann_file, transforms, return_masks=False, **kwargs):
        super(LvisDetection, self).__init__(img_folder, ann_file)
        self.ann_file = ann_file
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(LvisDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img = self._transforms(img)
        return img, target, idx
    
    def get_raw_image(self, idx):
        img, target = super(LvisDetection, self).__getitem__(idx)
        return img
    
    def categories(self):
        id2cat = {c["id"]: c for c in self.lvis.dataset["categories"]}
        all_cats = sorted(list(id2cat.keys()))
        categories = {}
        for l in list(all_cats):
            categories[l] = id2cat[l]['name']
        return categories