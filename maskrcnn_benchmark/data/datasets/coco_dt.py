"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

from .modulated_coco import ConvertCocoPolysToMask
from .tsv import ODTSVDataset
import random
from .od_to_grounding import convert_object_detection_to_grounding_optimized_for_od, check_for_positive_overflow, sanity_check_target_after_processing


class CocoDetectionTSV(ODTSVDataset):
    def __init__(self,
                 name,
                 yaml_file,
                 transforms,
                 return_tokens,
                 tokenizer,
                 extra_fields,
                 random_sample_negative=-1,
                 add_detection_prompt=False,
                 add_detection_prompt_advanced=False,
                 use_od_data_aug=False,
                 control_probabilities={},
                 disable_shuffle=False,
                 prompt_engineer_version="v2",
                 prompt_limit_negative=-1,
                 positive_question_probability=0.6,
                 negative_question_probability=0.8,
                 full_question_probability=0.5,
                 disable_clip_to_image=False,
                 separation_tokens=" ",
                 no_mask_for_od=False,
                 max_num_labels=-1,
                 max_query_len=256,
                 **kwargs
                 ):
        super(CocoDetectionTSV, self).__init__(yaml_file, extra_fields, **kwargs)

        self._transforms = transforms
        self.name = name
        self.max_query_len = max_query_len
        self.prepare = ConvertCocoPolysToMask(
            return_masks=False,
            return_tokens=return_tokens,
            tokenizer=tokenizer,
            max_query_len=max_query_len
        )
        self.tokenizer = tokenizer

        self.control_probabilities = control_probabilities
        self.random_sample_negative = random_sample_negative
        self.add_detection_prompt = add_detection_prompt
        self.add_detection_prompt_advanced = add_detection_prompt_advanced
        self.use_od_data_aug = use_od_data_aug

        self.prompt_engineer_version = prompt_engineer_version
        self.prompt_limit_negative = prompt_limit_negative
        self.positive_question_probability = positive_question_probability
        self.negative_question_probability = negative_question_probability
        self.full_question_probability = full_question_probability
        self.separation_tokens = separation_tokens
        self.disable_clip_to_image = disable_clip_to_image
        self.disable_shuffle = disable_shuffle
        self.no_mask_for_od = no_mask_for_od
        self.max_num_labels = max_num_labels

    def __len__(self):
        return super(CocoDetectionTSV, self).__len__()

    def categories(self, no_background=True):
        categories = self.coco.dataset["categories"]
        label_list = {}
        for index, i in enumerate(categories):
            # assert(index + 1 == i["id"])
            if not no_background or (i["name"] != "__background__" and i['id'] != 0):
                label_list[i["id"]] = i["name"]
        return label_list

    def __getitem__(self, idx):
        # tgt is a BoxList
        img, target, _, scale = super(CocoDetectionTSV, self).__getitem__(idx)
        image_id = self.get_img_id(idx)
        restricted_negative_list = None

        if not self.disable_clip_to_image:
            target = target.clip_to_image(remove_empty=True)

        original_box_num = len(target)

        target, positive_caption_length = check_for_positive_overflow(target, self.ind_to_class, self.tokenizer, self.max_query_len-2) # leave some space for the special tokens

        if len(target) < original_box_num:
            print("WARNING: removed {} boxes due to positive caption overflow".format(original_box_num - len(target)))

        annotations, caption, greenlight_span_for_masked_lm_objective, label_to_positions = convert_object_detection_to_grounding_optimized_for_od(
            target=target,
            image_id=image_id,
            ind_to_class=self.ind_to_class,
            disable_shuffle=self.disable_shuffle,
            add_detection_prompt=self.add_detection_prompt,
            add_detection_prompt_advanced=self.add_detection_prompt_advanced,
            random_sample_negative=self.random_sample_negative,
            control_probabilities=self.control_probabilities,
            restricted_negative_list=restricted_negative_list,
            separation_tokens=self.separation_tokens,
            max_num_labels=self.max_num_labels,
            positive_caption_length=positive_caption_length,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_query_len-2
        )

        # assert(len(self.tokenizer.tokenize(caption)) <= self.max_query_len-2)

        # print(caption)
        anno = {"image_id": image_id, "annotations": annotations, "caption": caption, "label_to_positions": label_to_positions}
        anno["greenlight_span_for_masked_lm_objective"] = greenlight_span_for_masked_lm_objective

        if self.no_mask_for_od:
            anno["greenlight_span_for_masked_lm_objective"].append((-1, -1, -1))

        img, anno = self.prepare(img, anno, box_format="xyxy")

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        
        # add additional property
        for ann in anno:
            target.add_field(ann, anno[ann])

        sanity_check_target_after_processing(target)

        return img, target, idx

    def get_raw_image(self, idx):
        image, *_ = super(CocoDetectionTSV, self).__getitem__(idx)
        return image

    def get_img_id(self, idx):
        line_no = self.get_line_no(idx)
        if self.label_tsv is not None:
            row = self.label_tsv.seek(line_no)
            img_id = row[0]
            try:
                return int(img_id)
            except:
                return idx
