# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import collections
import json
import os.path as op

import numpy as np
import torch

from .tsv import TSVYamlDataset, find_file_path_in_yaml
from .box_label_loader import BoxLabelLoader
from maskrcnn_benchmark.data.datasets.coco_dt import CocoDetectionTSV


class VGDetectionTSV(CocoDetectionTSV):
    pass


def sort_key_by_val(dic):
    sorted_dic = sorted(dic.items(), key=lambda kv: kv[1])
    return [kv[0] for kv in sorted_dic]


def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:, 2] - gt_boxes[:, 0] + 1) *
                     (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)).view(1, K)

    anchors_area = ((anchors[:, 2] - anchors[:, 0] + 1) *
                    (anchors[:, 3] - anchors[:, 1] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:, :, 2], query_boxes[:, :, 2]) -
          torch.max(boxes[:, :, 0], query_boxes[:, :, 0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:, :, 3], query_boxes[:, :, 3]) -
          torch.max(boxes[:, :, 1], query_boxes[:, :, 1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps


# VG data loader for Danfei Xu's Scene graph focused format.
# todo: if ordering of classes, attributes, relations changed
# todo make sure to re-write the obj_classes.txt/rel_classes.txt files

def _box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations.
    If no overlapping boxes, use all of them."""
    overlaps = bbox_overlaps(boxes, boxes).numpy() > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes


class VGTSVDataset(TSVYamlDataset):
    """
    Generic TSV dataset format for Object Detection.
    """

    def __init__(self, yaml_file, extra_fields=None, transforms=None,
                 is_load_label=True, filter_duplicate_rels=True,
                 relation_on=False, cv2_output=False, **kwargs):
        if extra_fields is None:
            extra_fields = []
        self.transforms = transforms
        self.is_load_label = is_load_label
        self.relation_on = relation_on
        super(VGTSVDataset, self).__init__(yaml_file, cv2_output=cv2_output)

        ignore_attrs = self.cfg.get("ignore_attrs", None)
        # construct those maps
        jsondict_file = find_file_path_in_yaml(self.cfg.get("jsondict", None), self.root)
        jsondict = json.load(open(jsondict_file, 'r'))

        # self.linelist_file
        if 'train' in op.basename(self.linelist_file):
            self.split = "train"
        elif 'test' in op.basename(self.linelist_file) \
                or 'val' in op.basename(self.linelist_file) \
                or 'valid' in op.basename(self.linelist_file):
            self.split = "test"
        else:
            raise ValueError("Split must be one of [train, test], but get {}!".format(self.linelist_file))
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'

        self.class_to_ind = jsondict['label_to_idx']
        self.ind_to_class = jsondict['idx_to_label']
        self.class_to_ind['__background__'] = 0
        self.ind_to_class['0'] = '__background__'
        self.classes = sort_key_by_val(self.class_to_ind)
        assert (all([self.classes[i] == self.ind_to_class[str(i)] for i in range(len(self.classes))]))

        # writing obj classes to disk for Neural Motif model building.
        obj_classes_out_fn = op.splitext(self.label_file)[0] + ".obj_classes.txt"
        if not op.isfile(obj_classes_out_fn):
            with open(obj_classes_out_fn, 'w') as f:
                for item in self.classes:
                    f.write("%s\n" % item)

        self.attribute_to_ind = jsondict['attribute_to_idx']
        self.ind_to_attribute = jsondict['idx_to_attribute']
        self.attribute_to_ind['__no_attribute__'] = 0
        self.ind_to_attribute['0'] = '__no_attribute__'
        self.attributes = sort_key_by_val(self.attribute_to_ind)
        assert (all([self.attributes[i] == self.ind_to_attribute[str(i)] for i in range(len(self.attributes))]))

        self.relation_to_ind = jsondict['predicate_to_idx']
        self.ind_to_relation = jsondict['idx_to_predicate']
        self.relation_to_ind['__no_relation__'] = 0
        self.ind_to_relation['0'] = '__no_relation__'
        self.relations = sort_key_by_val(self.relation_to_ind)
        assert (all([self.relations[i] == self.ind_to_relation[str(i)] for i in range(len(self.relations))]))

        # writing rel classes to disk for Neural Motif Model building.
        rel_classes_out_fn = op.splitext(self.label_file)[0] + '.rel_classes.txt'
        if not op.isfile(rel_classes_out_fn):
            with open(rel_classes_out_fn, 'w') as f:
                for item in self.relations:
                    f.write("%s\n" % item)

        # label map: minus one because we will add one in BoxLabelLoader
        self.labelmap = {key: val - 1 for key, val in self.class_to_ind.items()}
        labelmap_file = find_file_path_in_yaml(self.cfg.get("labelmap_dec"), self.root)
        # self.labelmap_dec = load_labelmap_file(labelmap_file)
        if self.is_load_label:
            self.label_loader = BoxLabelLoader(
                labelmap=self.labelmap,
                extra_fields=extra_fields,
                ignore_attrs=ignore_attrs
            )

        # get frequency prior for relations
        if self.relation_on:
            self.freq_prior_file = op.splitext(self.label_file)[0] + ".freq_prior.npy"
            if self.split == 'train' and not op.exists(self.freq_prior_file):
                print("Computing frequency prior matrix...")
                fg_matrix, bg_matrix = self._get_freq_prior()
                prob_matrix = fg_matrix.astype(np.float32)
                prob_matrix[:, :, 0] = bg_matrix
                prob_matrix[:, :, 0] += 1
                prob_matrix /= np.sum(prob_matrix, 2)[:, :, None]
                np.save(self.freq_prior_file, prob_matrix)

    def _get_freq_prior(self, must_overlap=False):
        fg_matrix = np.zeros((
            len(self.classes),
            len(self.classes),
            len(self.relations)
        ), dtype=np.int64)

        bg_matrix = np.zeros((
            len(self.classes),
            len(self.classes),
        ), dtype=np.int64)

        for ex_ind in range(self.__len__()):
            target = self.get_groundtruth(ex_ind)
            gt_classes = target.get_field('labels').numpy()
            gt_relations = target.get_field('relation_labels').numpy()
            gt_boxes = target.bbox

            # For the foreground, we'll just look at everything
            try:
                o1o2 = gt_classes[gt_relations[:, :2]]
                for (o1, o2), gtr in zip(o1o2, gt_relations[:, 2]):
                    fg_matrix[o1, o2, gtr] += 1

                # For the background, get all of the things that overlap.
                o1o2_total = gt_classes[np.array(
                    _box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
                for (o1, o2) in o1o2_total:
                    bg_matrix[o1, o2] += 1
            except IndexError as e:
                assert len(gt_relations) == 0

            if ex_ind % 20 == 0:
                print("processing {}/{}".format(ex_ind, self.__len__()))

        return fg_matrix, bg_matrix

    def relation_loader(self, relation_triplets, target):
        # relation_triplets [list of tuples]: M*3
        # target: BoxList from label_loader
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            all_rel_sets = collections.defaultdict(list)
            for (o0, o1, r) in relation_triplets:
                all_rel_sets[(o0, o1)].append(r)
            relation_triplets = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]

        # get M*M pred_labels
        relations = torch.zeros([len(target), len(target)], dtype=torch.int64)
        for i in range(len(relation_triplets)):
            subj_id = relation_triplets[i][0]
            obj_id = relation_triplets[i][1]
            pred = relation_triplets[i][2]
            relations[subj_id, obj_id] = int(pred)

        relation_triplets = torch.tensor(relation_triplets)
        target.add_field("relation_labels", relation_triplets)
        target.add_field("pred_labels", relations)
        return target

    def get_target_from_annotations(self, annotations, img_size, idx):
        if self.is_load_label and annotations:
            target = self.label_loader(annotations['objects'], img_size)
            # make sure no boxes are removed
            assert (len(annotations['objects']) == len(target))
            if self.split in ["val", "test"]:
                # add the difficult field
                target.add_field("difficult", torch.zeros(len(target), dtype=torch.int32))
            # load relations
            if self.relation_on:
                target = self.relation_loader(annotations["relations"], target)
            return target

    def get_groundtruth(self, idx, call=False):
        # similar to __getitem__ but without transform
        img = self.get_image(idx)
        if self.cv2_output:
            img_size = img.shape[:2][::-1]  # h, w -> w, h
        else:
            img_size = img.size  # w, h
        annotations = self.get_annotations(idx)
        target = self.get_target_from_annotations(annotations, img_size, idx)
        if call:
            return img, target, annotations
        else:
            return target

    def apply_transforms(self, img, target=None):
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def map_class_id_to_class_name(self, class_id):
        return self.classes[class_id]

    def map_attribute_id_to_attribute_name(self, attribute_id):
        return self.attributes[attribute_id]

    def map_relation_id_to_relation_name(self, relation_id):
        return self.relations[relation_id]
