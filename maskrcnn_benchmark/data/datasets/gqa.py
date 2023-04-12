import json
from pathlib import Path

import torch
import torchvision

from .modulated_coco import ConvertCocoPolysToMask, ModulatedDataset


class GQADataset(ModulatedDataset):
    pass


class GQAQuestionAnswering(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, return_tokens, tokenizer, ann_folder):
        super(GQAQuestionAnswering, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, return_tokens, tokenizer=tokenizer)
        with open(ann_folder / "gqa_answer2id.json", "r") as f:
            self.answer2id = json.load(f)
        with open(ann_folder / "gqa_answer2id_by_type.json", "r") as f:
            self.answer2id_by_type = json.load(f)
        self.type2id = {"obj": 0, "attr": 1, "rel": 2, "global": 3, "cat": 4}

    def __getitem__(self, idx):
        img, target = super(GQAQuestionAnswering, self).__getitem__(idx)
        image_id = self.ids[idx]
        coco_img = self.coco.loadImgs(image_id)[0]
        caption = coco_img["caption"]
        dataset_name = coco_img["dataset_name"]
        questionId = coco_img["questionId"]
        target = {"image_id": image_id, "annotations": target, "caption": caption}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        target["dataset_name"] = dataset_name
        target["questionId"] = questionId

        if coco_img["answer"] not in self.answer2id:
            answer = "unknown"
        else:
            answer = coco_img["answer"]

        target["answer"] = torch.as_tensor(self.answer2id[answer], dtype=torch.long)
        target["answer_type"] = torch.as_tensor(self.type2id[coco_img["question_type"]], dtype=torch.long)

        if coco_img["answer"] not in self.answer2id_by_type["answer_attr"]:
            answer = "unknown"
        else:
            answer = coco_img["answer"]
        target["answer_attr"] = torch.as_tensor(
            self.answer2id_by_type["answer_attr"][answer] if coco_img["question_type"] == "attr" else -100,
            dtype=torch.long,
        )

        if coco_img["answer"] not in self.answer2id_by_type["answer_global"]:
            answer = "unknown"
        else:
            answer = coco_img["answer"]
        target["answer_global"] = torch.as_tensor(
            self.answer2id_by_type["answer_global"][answer] if coco_img["question_type"] == "global" else -100,
            dtype=torch.long,
        )

        if coco_img["answer"] not in self.answer2id_by_type["answer_rel"]:
            answer = "unknown"
        else:
            answer = coco_img["answer"]
        target["answer_rel"] = torch.as_tensor(
            self.answer2id_by_type["answer_rel"][answer] if coco_img["question_type"] == "rel" else -100,
            dtype=torch.long,
        )

        if coco_img["answer"] not in self.answer2id_by_type["answer_cat"]:
            answer = "unknown"
        else:
            answer = coco_img["answer"]
        target["answer_cat"] = torch.as_tensor(
            self.answer2id_by_type["answer_cat"][answer] if coco_img["question_type"] == "cat" else -100,
            dtype=torch.long,
        )

        if coco_img["answer"] not in self.answer2id_by_type["answer_obj"]:
            answer = "unknown"
        else:
            answer = coco_img["answer"]
        target["answer_obj"] = torch.as_tensor(
            self.answer2id_by_type["answer_obj"][answer] if coco_img["question_type"] == "obj" else -100,
            dtype=torch.long,
        )
        return img, target
