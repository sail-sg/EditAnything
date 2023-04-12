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

class CaptionTSV(TSVYamlDataset):
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
                 **kwargs
                 ):
        super(CaptionTSV, self).__init__(yaml_file, None, replace_clean_label)
        self.yaml_file = yaml_file
        self._transforms = transforms
        self.max_query_len = max_query_len
        self.prepare = ConvertCocoPolysToMask(return_masks=return_masks,
                                              return_tokens=return_tokens,
                                              tokenizer=tokenizer,
                                              max_query_len=max_query_len)
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
        try:
            self.rank = dist.get_rank()
        except:
            self.rank = 0

    def __len__(self):
        return super(CaptionTSV, self).__len__()

    def pack_caption(self, positive_caption, negative_captions, original_tokens_positive):
        if len(negative_captions) == 0:
            return positive_caption, original_tokens_positive, [(0, len(positive_caption))]
        if self.safeguard_positive_caption:
            length_of_each_caption = []
            for caption in negative_captions + [positive_caption]:
                tokenized = self.tokenizer(caption, return_tensors="pt")
                length_of_each_caption.append(tokenized.input_ids.size(-1))
            max_length = self.max_query_len - length_of_each_caption[-1]
            indexes = list(range(len(negative_captions)))
            random.shuffle(indexes)
            new_caption_list = [positive_caption]
            for i in indexes:
                if length_of_each_caption[i] < max_length:
                    new_caption_list.append(negative_captions[i])
                    max_length -= length_of_each_caption[i]
        else:
            new_caption_list = [positive_caption] + negative_captions
        random.shuffle(new_caption_list)

        new_caption = ''

        for i in new_caption_list:
            if i == positive_caption:
                start_position = len(new_caption)
            new_caption += i
            if not i.endswith("."):
                new_caption += "."
            new_caption += " "

        # shift the token positions the boxes are aligned to
        for index, i in enumerate(original_tokens_positive):
            original_tokens_positive[index] = [tuple(j) for j in i]
        for i in original_tokens_positive:
            for index, j in enumerate(i):
                i[index] = (j[0] + start_position, j[1] + start_position)

        return new_caption, original_tokens_positive, [(start_position, start_position + len(positive_caption))]

    def __get_negative_captions__(self, idx, negative_size=7):
        negative_captions = []
        for i in range(negative_size):
            img, anno, _, scale = super(CaptionTSV, self).__getitem__(np.random.choice(len(self)))
            caption = anno["caption"]
            negative_captions.append(caption)

        return negative_captions

    def __getitem__(self, idx):
        try:
            img, anno, _, scale = super(CaptionTSV, self).__getitem__(idx)
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
                '''
                An example
                {'img_h': 1154, 'img_w': 1600, 'caption': 'xxx', 'tokens_positive': [[[47, 50], [51, 53], [54, 59]], [[32, 35], [36, 41]], [[32, 35], [36, 41]], [[0, 3], [3, 6], [6, 10], [11, 16], [17, 19], [20, 23]], [[32, 35], [36, 41]], [[32, 35], [36, 41]]], 'bboxes': [[7.344961166381836, 10.479412078857422, 1592.2679443359375, 1090.0028076171875], [950.32861328125, 346.572021484375, 1333.2373046875, 679.3215942382812], [927.44140625, 342.7712707519531, 1389.833984375, 719.5758666992188], [90.48786163330078, 363.67572021484375, 1381.8631591796875, 1078.687744140625], [122.84217071533203, 422.6786193847656, 507.845703125, 667.2651977539062], [80.62384033203125, 416.500244140625, 563.1666259765625, 734.603271484375]], 'scores': [0.7966700196266174, 0.8952182531356812, 0.8186006546020508, 0.9995516538619995, 0.8021856546401978, 0.8923134803771973]}
                '''
                if len(anno["bboxes"]) < self.caption_min_box:  # Retry triggered!
                    return self[np.random.choice(len(self))]

                if self.caption_format_version == "v2":
                    anno = self.convert_anno_from_v2_to_v1(anno)

                try:
                    if self.further_screen:
                        conf = self.caption_conf
                        nms_thre = self.caption_nms

                        bboxes = torch.as_tensor(anno["bboxes"]).float()
                        scores = torch.as_tensor(anno["scores"])
                        tokens_positive = anno["tokens_positive"]

                        # print("\n\n\n\n tokens_positive in original data", tokens_positive)

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

                    caption = anno["caption"]
                    # print("original caption", caption)
                    empty_everything = False
                    if self.sample_negative_for_grounding_data != -1:
                        if random.random() < self.sample_negative_for_grounding_data:
                            empty_everything = True

                    if empty_everything:
                        caption = self.__get_negative_captions__(idx, negative_size=1)[0]

                    if self.pack_random_caption_number != 0:
                        if self.random_pack_prob != -1.0:
                            if random.random() < self.no_random_pack_probability:
                                negative_pack_number = 0
                            elif random.random() < self.random_pack_prob:
                                negative_pack_number = self.pack_random_caption_number
                            else:
                                negative_pack_number = np.random.choice(self.pack_random_caption_number)
                        else:
                            negative_pack_number = self.pack_random_caption_number

                        negative_captions = self.__get_negative_captions__(idx, negative_size=negative_pack_number)

                        caption, anno["tokens_positive"], greenlight_span_for_masked_lm_objective = self.pack_caption(
                            caption, negative_captions, anno["tokens_positive"])
                    else:
                        greenlight_span_for_masked_lm_objective = [(0, len(caption))]

                    if not self.mlm_obj_for_only_positive:
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

                except:
                    return self[np.random.choice(len(self))]

                anno = new_anno
                if empty_everything:
                    anno = []

            annotations = {"image_id": idx, "annotations": anno, "caption": caption}
            annotations["greenlight_span_for_masked_lm_objective"] = greenlight_span_for_masked_lm_objective
            img, annotations = self.prepare(img, annotations, box_format="xyxy")

            if self._transforms is not None:
                img, target = self._transforms(img, target)

            # add additional property
            for ann in annotations:
                target.add_field(ann, annotations[ann])
        except:
            print("Outter Retry triggered!!")
            return self[np.random.choice(len(self))]

        sanity_check_target_after_processing(target)
        
        return img, target, idx

    def convert_anno_from_v2_to_v1(self, anno):
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
        image, *_ = super(CaptionTSV, self).__getitem__(idx)
        return image

    def get_img_id(self, idx):
        line_no = self.get_line_no(idx)
        if self.label_tsv is not None:
            row = self.label_tsv.seek(line_no)
            img_id = row[0]
            return img_id
