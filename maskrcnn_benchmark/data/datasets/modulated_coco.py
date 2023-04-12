import logging
import os
import os.path
from PIL import Image

import torch
import torchvision
import torch.utils.data as data
from pycocotools import mask as coco_mask

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.data.datasets.coco import has_valid_annotation
from .od_to_grounding import convert_od_to_grounding_simple, check_for_positive_overflow, sanity_check_target_after_processing, convert_object_detection_to_grounding_optimized_for_od


class CocoGrounding(torchvision.datasets.CocoDetection):
    def __init__(self,
                 img_folder,
                 ann_file,
                 transforms,
                 return_masks,
                 return_tokens,
                 is_train=False,
                 tokenizer=None,
                 disable_shuffle=False,
                 add_detection_prompt=False,
                 one_hot=False,
                 disable_clip_to_image=False,
                 no_minus_one_for_one_hot=False,
                 separation_tokens=" ",
                 few_shot=0,
                 no_mask_for_od=False,
                 override_category=None,
                 use_caption_prompt=False,
                 caption_prompt=None,
                 max_query_len=256,
                 special_safeguard_for_coco_grounding=False,
                 random_sample_negative=-1,
                 **kwargs
                 ):
        super(CocoGrounding, self).__init__(img_folder, ann_file)
        self.ids = sorted(self.ids)

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
            # cats_freq = [few_shot]*len(self.coco.cats.keys())
            cats_freq = [few_shot]*max(list(self.coco.cats.keys()))
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



        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        if override_category is not None:
            self.coco.dataset["categories"] = override_category
        self.use_caption_prompt = use_caption_prompt
        self.caption_prompt = caption_prompt
        self.special_safeguard_for_coco_grounding = special_safeguard_for_coco_grounding
        self.random_sample_negative = random_sample_negative
        self.ind_to_class = self.categories(no_background=False)
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms
        self.max_query_len = max_query_len
        self.prepare = ConvertCocoPolysToMask(False, return_tokens, tokenizer=tokenizer, max_query_len=max_query_len)
        self.tokenizer = tokenizer
        self.is_train = is_train

        self.ind_to_class = self.categories(no_background=False)

        self.disable_shuffle = disable_shuffle
        self.add_detection_prompt = add_detection_prompt
        self.one_hot = one_hot
        self.no_minus_one_for_one_hot = no_minus_one_for_one_hot

        self.disable_clip_to_image = disable_clip_to_image
        self.separation_tokens = separation_tokens
        self.no_mask_for_od = no_mask_for_od
        self.return_masks = return_masks

    def categories(self, no_background=True):
        categories = self.coco.dataset["categories"]
        label_list = {}
        for index, i in enumerate(categories):
            # assert(index + 1 == i["id"])
            if not no_background or (i["name"] != "__background__" and i['id'] != 0):
                label_list[self.json_category_id_to_contiguous_id[i["id"]]] = i["name"]
        return label_list

    def get_box_mask(self, rect, img_size, mode="poly"):
        assert mode=="poly", "Only support poly mask right now!"
        x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
        return [[x1, y1, x1, y2, x2, y2, x2, y1]]

    def __getitem__(self, idx):
        img, tgt = super(CocoGrounding, self).__getitem__(idx)
        image_id = self.ids[idx]
        tgt = [obj for obj in tgt if obj["iscrowd"] == 0]
        boxes = [obj["bbox"] for obj in tgt]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
        classes = [obj["category_id"] for obj in tgt]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if self.return_masks:
            masks = []
            is_box_mask = []
            for obj, bbox in zip(tgt, target.bbox):
                if "segmentation" in obj:
                    masks.append(obj["segmentation"])
                    is_box_mask.append(0)
                else:
                    masks.append(self.get_box_mask(bbox, img.size, mode="poly"))
                    is_box_mask.append(1)
            masks = SegmentationMask(masks, img.size, mode="poly")
            is_box_mask = torch.tensor(is_box_mask)
            target.add_field("masks", masks)
            target.add_field("is_box_mask", is_box_mask)
        
        if not self.disable_clip_to_image:
            target = target.clip_to_image(remove_empty=True)
        
        if self.special_safeguard_for_coco_grounding:
            # Intended for LVIS
            assert(not self.use_caption_prompt)

            original_box_num = len(target)
            target, positive_caption_length = check_for_positive_overflow(target, self.ind_to_class, self.tokenizer, self.max_query_len-2) # leave some space for the special tokens
            if len(target) < original_box_num:
                print("WARNING: removed {} boxes due to positive caption overflow".format(original_box_num - len(target)))

            annotations, caption, greenlight_span_for_masked_lm_objective, label_to_positions = convert_object_detection_to_grounding_optimized_for_od(
                target=target,
                image_id=image_id,
                ind_to_class=self.ind_to_class,
                disable_shuffle=self.disable_shuffle,
                add_detection_prompt=False,
                add_detection_prompt_advanced=False,
                random_sample_negative=self.random_sample_negative,
                control_probabilities=(0.0, 0.0, 1.0, 0.0), # always try to add a lot of negatives
                restricted_negative_list=None,
                separation_tokens=self.separation_tokens,
                max_num_labels=-1,
                positive_caption_length=positive_caption_length,
                tokenizer=self.tokenizer,
                max_seq_length=self.max_query_len-2
            )
        else:
            # Intended for COCO / ODinW
            annotations, caption, greenlight_span_for_masked_lm_objective = convert_od_to_grounding_simple(
                target=target,
                image_id=image_id,
                ind_to_class=self.ind_to_class,
                disable_shuffle=self.disable_shuffle,
                add_detection_prompt=self.add_detection_prompt,
                separation_tokens=self.separation_tokens,
                caption_prompt=self.caption_prompt if self.use_caption_prompt else None,
            )

        anno = {"image_id": image_id, "annotations": annotations, "caption": caption}
        anno["greenlight_span_for_masked_lm_objective"] = greenlight_span_for_masked_lm_objective
        if self.no_mask_for_od:
            anno["greenlight_span_for_masked_lm_objective"].append((-1, -1, -1))
        img, anno = self.prepare(img, anno, box_format="xyxy")

        # for equivalence check
        if self.one_hot:
            logging.info("using one hot for equivalence check.")
            one_hot_map = torch.zeros_like(anno["positive_map"], dtype=torch.float)
            text_mask = torch.zeros(anno["positive_map"].shape[1], dtype=torch.int64)
            # create one hot mapping
            for ii, cls in enumerate(classes):
                if self.no_minus_one_for_one_hot:
                    one_hot_map[ii, cls] = 1.0
                else:
                    one_hot_map[ii, cls - 1] = 1.0
            if self.no_minus_one_for_one_hot:
                text_mask[:] = 1
            else:
                text_mask[:len(self.ind_to_class)] = 1
            anno["positive_map"] = one_hot_map
            anno["text_mask"] = text_mask

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # add additional property
        for ann in anno:
            target.add_field(ann, anno[ann])
        
        sanity_check_target_after_processing(target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data


class ModulatedDataset(torchvision.datasets.CocoDetection):
    def __init__(self,
                 img_folder,
                 ann_file,
                 transforms,
                 return_masks,
                 return_tokens,
                 is_train=False,
                 tokenizer=None,
                 disable_clip_to_image=False,
                 no_mask_for_gold=False,
                 max_query_len=256,
                 **kwargs):
        super(ModulatedDataset, self).__init__(img_folder, ann_file)
        self.ids = sorted(self.ids)

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

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms
        self.max_query_len = max_query_len
        self.prepare = ConvertCocoPolysToMask(return_masks, return_tokens, tokenizer=tokenizer, max_query_len=max_query_len)
        self.is_train = is_train
        self.disable_clip_to_image = disable_clip_to_image
        self.no_mask_for_gold = no_mask_for_gold

    def __getitem__(self, idx):
        img, target = super(ModulatedDataset, self).__getitem__(idx)
        image_id = self.ids[idx]
        coco_img = self.coco.loadImgs(image_id)[0]
        caption = coco_img["caption"]
        dataset_name = coco_img["dataset_name"] if "dataset_name" in coco_img else None
        anno = {"image_id": image_id, "annotations": target, "caption": caption}

        # This dataset is used for Flickr & Mixed, so the sequence is maskable
        anno["greenlight_span_for_masked_lm_objective"] = [(0, len(caption))]
        if self.no_mask_for_gold:
            anno["greenlight_span_for_masked_lm_objective"].append((-1, -1, -1))
        img, anno = self.prepare(img, anno)

        # convert to BoxList (bboxes, labels)
        boxes = torch.as_tensor(anno["boxes"]).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xyxy")
        classes = anno["labels"]
        target.add_field("labels", classes)
        if self.prepare.return_masks:
            target.add_field("masks", anno.pop("masks"))
            target.add_field("is_box_mask", anno.pop("is_box_mask"))
        if not self.disable_clip_to_image:
            num_boxes = len(target.bbox)
            target = target.clip_to_image(remove_empty=True)
            assert num_boxes == len(target.bbox), "Box got removed in MixedDataset!!!"

        # Check if bboxes are correct
        # draw = ImageDraw.Draw(img)
        # boxes = target.bbox
        # for box in boxes:
        #     draw.rectangle([box[0], box[1], box[2], box[3]])
        # img.save('OUTPUT/images/{}.jpg'.format(idx))

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # add additional property
        for ann in anno:
            target.add_field(ann, anno[ann])

        target.add_field("dataset_name", dataset_name)
        for extra_key in ["sentence_id", "original_img_id", "original_id", "task_id"]:
            if extra_key in coco_img:
                target.add_field(extra_key, coco_img[extra_key])

        if "tokens_positive_eval" in coco_img and not self.is_train:
            tokenized = self.prepare.tokenizer(caption, return_tensors="pt")
            target.add_field("positive_map_eval", create_positive_map(tokenized, coco_img["tokens_positive_eval"]))
            target.add_field("nb_eval", len(target.get_field("positive_map_eval")))

        sanity_check_target_after_processing(target)
        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data


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


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, return_tokens=False, tokenizer=None, max_query_len=256):
        self.return_masks = return_masks
        self.return_tokens = return_tokens
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len

    def get_box_mask(self, rect, img_size, mode="poly"):
        assert mode=="poly", "Only support poly mask right now!"
        x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
        return [[x1, y1, x1, y2, x2, y2, x2, y1]]

    def __call__(self, image, target, ignore_box_screen=False, box_format="xywh"):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        caption = target["caption"] if "caption" in target else None
        label_to_positions = target.get("label_to_positions", {})

        greenlight_span_for_masked_lm_objective = target.get("greenlight_span_for_masked_lm_objective", None)

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if box_format == "xywh":
            boxes[:, 2:] += boxes[:, :2] - 1  # TO_REMOVE = 1
            boxes[:, 0::2].clamp_(min=0, max=w-1)  # TO_REMOVE = 1
            boxes[:, 1::2].clamp_(min=0, max=h-1)  # TO_REMOVE = 1

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            masks = []
            is_box_mask = []
            for obj, bbox in zip(anno, boxes):
                if "segmentation" in obj:
                    masks.append(obj["segmentation"])
                    is_box_mask.append(0)
                else:
                    masks.append(self.get_box_mask(bbox, image.size, mode='poly'))
                    is_box_mask.append(1)
            masks = SegmentationMask(masks, image.size, mode='poly')
            is_box_mask = torch.tensor(is_box_mask)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        isfinal = None
        if anno and "isfinal" in anno[0]:
            isfinal = torch.as_tensor([obj["isfinal"] for obj in anno], dtype=torch.float)

        tokens_positive = [] if self.return_tokens else None
        if self.return_tokens and anno and "tokens" in anno[0]:
            tokens_positive = [obj["tokens"] for obj in anno]
        elif self.return_tokens and anno and "tokens_positive" in anno[0]:
            tokens_positive = [obj["tokens_positive"] for obj in anno]

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
            is_box_mask = is_box_mask[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if caption is not None:
            target["caption"] = caption
        if self.return_masks:
            target["masks"] = masks
            target["is_box_mask"] = is_box_mask
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        if tokens_positive is not None:
            target["tokens_positive"] = []

            for i, k in enumerate(keep):
                if k or ignore_box_screen:
                    target["tokens_positive"].append(tokens_positive[i])

        if isfinal is not None:
            target["isfinal"] = isfinal

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if self.return_tokens and self.tokenizer is not None:
            if not ignore_box_screen:
                assert len(target["boxes"]) == len(target["tokens_positive"])
            tokenized = self.tokenizer(caption, return_tensors="pt",
                max_length=self.max_query_len,
                truncation=True)
            target["positive_map"] = create_positive_map(tokenized, target["tokens_positive"])
            target['greenlight_map'] = create_greenlight_map(greenlight_span_for_masked_lm_objective,tokenized)
            target["positive_map_for_od_labels"] = create_positive_map_for_od_labels(tokenized, label_to_positions)

        original_od_label = []
        for obj in anno:
            original_od_label.append(
                obj.get("original_od_label", -10))  # NOTE: The padding value has to be not the same as -1 or -100
        target["original_od_label"] = torch.as_tensor(original_od_label)

        return image, target

def create_greenlight_map(tok_list, tokenized):
    # An example tok_list:
    # [(0, 5), (10, 13), (-1, -1, -1)]
    # The last one is a special indicator..

    greenlight_map = torch.zeros(256, dtype=torch.float)
    for item in tok_list:
        if len(item) != 2:
            assert(len(item) == 3)
            # Make everything unmakable
            greenlight_map[:] = -1
            break

        beg, end = item
        beg_pos = tokenized.char_to_token(beg)
        end_pos = tokenized.char_to_token(end - 1)
        if beg_pos is None:
            try:
                beg_pos = tokenized.char_to_token(beg + 1)
                if beg_pos is None:
                    beg_pos = tokenized.char_to_token(beg + 2)
            except:
                beg_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end - 2)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end - 3)
            except:
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue

        assert beg_pos is not None and end_pos is not None
        greenlight_map[beg_pos: end_pos + 1].fill_(1)
    return greenlight_map


def create_positive_map_for_od_labels(tokenized, label_to_positions):
    """construct a map such that positive_map[i] = j, where j is the object detection label of the token i"""
    """
    {3: [1: 5)}
    256 : -1 3 3 3 3 -1 .. 8 8 ..
    the woman in the garden
    -1 -1 -1 -1 -1
    """
    positive_map = torch.ones(256, dtype=torch.float) * -1  # -1 means no match
    keys = list(label_to_positions.keys())
    for j, key in enumerate(keys):
        tok_list = label_to_positions[key]
        # one label only mapps to one location
        beg, end = tok_list
        beg_pos = tokenized.char_to_token(beg)
        end_pos = tokenized.char_to_token(end - 1)
        if beg_pos is None:
            try:
                beg_pos = tokenized.char_to_token(beg + 1)
                if beg_pos is None:
                    beg_pos = tokenized.char_to_token(beg + 2)
            except:
                beg_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end - 2)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end - 3)
            except:
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        assert beg_pos is not None and end_pos is not None
        positive_map[beg_pos: end_pos + 1].fill_(key)
    return positive_map


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def create_positive_map(tokenized, tokens_positive):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos: end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


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
