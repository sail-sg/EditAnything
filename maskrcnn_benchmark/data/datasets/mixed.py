import os
import os.path
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList

from PIL import Image, ImageDraw
from torchvision.datasets.vision import VisionDataset

from .modulated_coco import ConvertCocoPolysToMask, has_valid_annotation


class CustomCocoDetection(VisionDataset):
    """Coco-style dataset imported from TorchVision.
        It is modified to handle several image sources

    Args:
        root_coco (string): Path to the coco images
        root_vg (string): Path to the vg images
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
            self,
            root_coco: str,
            root_vg: str,
            annFile: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        super(CustomCocoDetection, self).__init__(root_coco, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

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

        self.root_coco = root_coco
        self.root_vg = root_vg

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        img_info = coco.loadImgs(img_id)[0]
        path = img_info["file_name"]
        dataset = img_info["data_source"]

        cur_root = self.root_coco if dataset == "coco" else self.root_vg
        img = Image.open(os.path.join(cur_root, path)).convert("RGB")
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


class MixedDataset(CustomCocoDetection):
    """Same as the modulated detection dataset, except with multiple img sources"""

    def __init__(self,
                 img_folder_coco,
                 img_folder_vg,
                 ann_file,
                 transforms,
                 return_masks,
                 return_tokens,
                 tokenizer=None,
                 disable_clip_to_image=False,
                 no_mask_for_gold=False,
                 max_query_len=256,
                 **kwargs):
        super(MixedDataset, self).__init__(img_folder_coco, img_folder_vg, ann_file)
        self._transforms = transforms
        self.max_query_len = max_query_len
        self.prepare = ConvertCocoPolysToMask(return_masks, return_tokens, tokenizer=tokenizer, max_query_len=max_query_len)
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.disable_clip_to_image = disable_clip_to_image
        self.no_mask_for_gold = no_mask_for_gold

    def __getitem__(self, idx):
        img, target = super(MixedDataset, self).__getitem__(idx)

        image_id = self.ids[idx]
        caption = self.coco.loadImgs(image_id)[0]["caption"]
        anno = {"image_id": image_id, "annotations": target, "caption": caption}
        anno["greenlight_span_for_masked_lm_objective"] = [(0, len(caption))]
        if self.no_mask_for_gold:
            anno["greenlight_span_for_masked_lm_objective"].append((-1, -1, -1))

        img, anno = self.prepare(img, anno)

        # convert to BoxList (bboxes, labels)
        boxes = torch.as_tensor(anno["boxes"]).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xyxy")
        classes = anno["labels"]
        target.add_field("labels", classes)
        if not self.disable_clip_to_image:
            num_boxes = len(boxes)
            target = target.clip_to_image(remove_empty=True)
            assert len(target.bbox) == num_boxes, "Box removed in MixedDataset!!!"

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # add additional property
        for ann in anno:
            target.add_field(ann, anno[ann])

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
