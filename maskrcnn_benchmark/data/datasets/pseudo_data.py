import torch
import torch.distributed as dist
import time
from torchvision.ops import nms
import random
import numpy as np
from PIL import Image, ImageDraw
import pdb
from maskrcnn_benchmark.structures.bounding_box import BoxList
from .modulated_coco import ConvertCocoPolysToMask
from .tsv import ODTSVDataset, TSVYamlDataset
from .od_to_grounding import sanity_check_target_after_processing
from copy import deepcopy

class PseudoData(TSVYamlDataset):
    def __init__(self,
                 yaml_file,
                 transforms,
                 return_tokens,
                 return_masks,
                 tokenizer,
                 caption_min_box=1,
                 replace_clean_label=False,
                 further_screen=False,
                 caption_conf=0.5,
                 caption_nms=-1,
                 pack_random_caption_number=0,
                 inference_caption=False,
                 sample_negative_for_grounding_data=-1,
                 random_pack_prob=-1.0,
                 no_random_pack_probability=0.0,
                 safeguard_positive_caption=True,
                 mlm_obj_for_only_positive=False,
                 caption_format_version="v1",
                 local_debug=False,
                 max_query_len=256,
                 diver_box_for_vqa=False,
                 **kwargs
                 ):
        super(PseudoData, self).__init__(yaml_file, None, replace_clean_label)
        self.yaml_file = yaml_file
        self._transforms = transforms
        self.max_query_len = max_query_len
        self.prepare = ConvertCocoPolysToMask(return_masks=return_masks,
                                              return_tokens=return_tokens,
                                              tokenizer=tokenizer,
                                              max_query_len=max_query_len)
        self.diver_box_for_vqa = diver_box_for_vqa
        if "qa" in self.yaml_file:
            assert(self.diver_box_for_vqa) # must diver box
        self.tokenizer = tokenizer
        self.caption_min_box = caption_min_box
        self.replace_clean_label = replace_clean_label
        self.further_screen = further_screen
        self.pack_random_caption_number = pack_random_caption_number
        self.caption_format_version = caption_format_version

        self.caption_conf = caption_conf
        self.caption_nms = caption_nms
        self.inference_caption = inference_caption
        self.sample_negative_for_grounding_data = sample_negative_for_grounding_data
        self.random_pack_prob = random_pack_prob
        self.no_random_pack_probability = no_random_pack_probability
        self.safeguard_positive_caption = safeguard_positive_caption
        self.mlm_obj_for_only_positive = mlm_obj_for_only_positive
        self.local_debug = local_debug
        try:
            self.rank = dist.get_rank()
        except:
            self.rank = 0

    def __len__(self):
        return super(PseudoData, self).__len__()

    @staticmethod
    def check_for_overlap(range1, range2):
        if range1[0] > range2[1] or range2[0] > range1[1]:
            return False
        return True

    def divert_boxes(self, anno):
        # first get answer start and end
        answer_start = len(anno['text']) + 1 # +1 for the space
        answer_end = len(anno["caption"])

        question = anno["caption"][:answer_start] # get the question

        mask_start = len(question)
        # add the mask token
        mask_token = self.tokenizer.mask_token
        if mask_token is None:
            mask_token = 'answer'
        question += mask_token
        mask_end = len(question)

        # divert the box
        for i in range(len(anno["bboxes"])):
            # check over lap
            for j in range(len(anno["tokens_positive"][i])): 
                if self.check_for_overlap(anno["tokens_positive"][i][j], [answer_start, answer_end]):
                    # if overlap, then divert the box to the mask token
                    anno["tokens_positive"][i][j] = [mask_start, mask_end]
        
        anno["caption"] = question
        return question, anno

    def __getitem__(self, idx):
        img, anno, _, scale = super(PseudoData, self).__getitem__(idx)
        if self.inference_caption:
            caption = None
            if isinstance(anno, list):
                caption = anno[0]["caption"]  # inference mode for bing
                anno = []
            elif len(anno) == 1:
                caption = anno["caption"]  # inference mode for googlecc
                anno = []
            else:
                caption = " ".join(anno["captions"])
                anno = []
        else:
            if self.caption_format_version == "v2":
                anno = self.convert_anno_from_yiling_to_ours(anno)
            
            if self.further_screen:
                conf = self.caption_conf
                nms_thre = self.caption_nms

                bboxes = torch.as_tensor(anno["bboxes"]).float()
                scores = torch.as_tensor(anno["scores"])
                tokens_positive = anno["tokens_positive"]

                keep = scores > conf
                scores = scores[keep]
                bboxes = bboxes[keep]
                tokens_positive = [i for index, i in enumerate(tokens_positive) if keep[index]]

                assert (len(tokens_positive) == len(bboxes) == len(scores))

                if len(bboxes) < self.caption_min_box:  # Retry triggered!
                    return self[np.random.choice(len(self))]

                if nms_thre > 0:
                    keep = nms(boxes=bboxes, scores=scores, iou_threshold=nms_thre)
                    scores = scores[keep]
                    bboxes = bboxes[keep]
                    tokens_positive = [tokens_positive[i] for i in keep]
                    assert (len(tokens_positive) == len(bboxes) == len(scores))

                # Write back
                anno["bboxes"] = bboxes.tolist()
                anno["scores"] = scores.tolist()
                anno["tokens_positive"] = tokens_positive

            boxes = torch.as_tensor(anno["bboxes"])

            if len(boxes) < self.caption_min_box:  # Retry triggered!
                return self[np.random.choice(len(self))]

            target = BoxList(boxes, (anno["img_w"], anno["img_h"]), mode="xyxy")
            target = target.clip_to_image(remove_empty=True)

            if self.diver_box_for_vqa:
                caption, anno = self.divert_boxes(anno=anno) # will change caption and "tokens_positive"

            caption = anno["caption"]
            
            greenlight_span_for_masked_lm_objective = [(0, len(caption))]

            new_anno = []
            areas = target.area()
            for i in range(len(target)):
                new_anno_i = {}
                new_anno_i["area"] = areas[i]
                new_anno_i["iscrowd"] = 0
                new_anno_i["image_id"] = idx
                new_anno_i["category_id"] = 1  # following vg and others
                new_anno_i["id"] = None
                new_anno_i['bbox'] = target.bbox[i].numpy().tolist()
                new_anno_i["tokens_positive"] = anno["tokens_positive"][i]
                new_anno.append(new_anno_i)
            anno = new_anno

        annotations = {"image_id": idx, "annotations": anno, "caption": caption}
        annotations["greenlight_span_for_masked_lm_objective"] = greenlight_span_for_masked_lm_objective
        img, annotations = self.prepare(img, annotations, box_format="xyxy")

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # add additional property
        for ann in annotations:
            target.add_field(ann, annotations[ann])
        
        # This is the real image_id
        image_id = self.get_img_id(idx)
        # Can insert additional field into target if needed
       
        sanity_check_target_after_processing(target)
        
        return img, target, idx

    def convert_anno_from_yiling_to_ours(self, anno):
        flatterned_bboxes = []
        flatterned_tokens_positive = []
        flatterned_bboxes_scores = []
        for i in range(len(anno["bboxes"])):
            # i is the index for entity
            for j in range(len(anno["bboxes"][i])):
                # j is the index for each box
                flatterned_bboxes.append(anno["bboxes"][i][j])
                flatterned_tokens_positive.append(
                    anno["tokens_positive"][i])  # Assume this box corresponds to all the token_spans for this entity
                flatterned_bboxes_scores.append(anno["scores"][i][j])
        anno["bboxes"] = flatterned_bboxes
        anno["tokens_positive"] = flatterned_tokens_positive
        anno["scores"] = flatterned_bboxes_scores
        return anno

    def get_raw_image(self, idx):
        image, *_ = super(PseudoData, self).__getitem__(idx)
        return image

    def get_img_id(self, idx):
        line_no = self.get_line_no(idx)
        if self.label_tsv is not None:
            row = self.label_tsv.seek(line_no)
            img_id = row[0]
            return img_id
