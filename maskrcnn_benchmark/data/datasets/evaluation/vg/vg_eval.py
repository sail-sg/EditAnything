# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)
from __future__ import division

import os
from collections import OrderedDict
import numpy as np
import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList
from glip.maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, getUnionBBox


# inspired from Detectron
def evaluate_box_proposals(
    predictions, dataset, thresholds=None, area="all", limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))

        # deal with ground truth
        gt_boxes = dataset.get_groundtruth(image_id)
        # filter out the field "relations"
        gt_boxes = gt_boxes.copy_with_fields(['attributes', 'labels'])
        gt_areas = gt_boxes.area()

        if len(gt_boxes) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        _gt_overlaps = torch.zeros(len(gt_boxes))
        if len(prediction) == 0:
            gt_overlaps.append(_gt_overlaps)
            continue
        if "objectness" in prediction.extra_fields:
            inds = prediction.get_field("objectness").sort(descending=True)[1]
        elif "scores" in prediction.extra_fields:
            inds = prediction.get_field("scores").sort(descending=True)[1]
        else:
            raise ValueError("Neither objectness nor scores is in the extra_fields!")
        prediction = prediction[inds]

        if limit is not None and len(prediction) > limit:
            prediction = prediction[:limit]

        overlaps = boxlist_iou(prediction, gt_boxes)

        for j in range(min(len(prediction), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


class VGResults(object):
    METRICS = {
        "bbox": ["AP",],
        "segm": ["AP",],
        "box_proposal": ["AR@100",],
    }

    def __init__(self, iou_type, value):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert iou_type in allowed_types
        results = OrderedDict()
        results[iou_type] = OrderedDict([(metric, value) for metric in VGResults.METRICS[iou_type]])
        self.results = results


def do_vg_evaluation(dataset, predictions, output_folder, box_only, eval_attributes, logger, save_predictions=True):
    # TODO need to make the use_07_metric format available
    # for the user to choose
    # we use int for box_only. 0: False, 1: box for RPN, 2: box for object detection, 
    if box_only:
        if box_only == 1:
            limits = [100, 1000]
        elif box_only == 2:
            limits = [36, 99]
        else:
            raise ValueError("box_only can be either 0/1/2, but get {0}".format(box_only))
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        result = {}
        for area, suffix in areas.items():
            for limit in limits:
                logger.info("Evaluating bbox proposals@{:d}".format(limit))
                stats = evaluate_box_proposals(
                    predictions, dataset, area=area, limit=limit
                )
                key_ar = "AR{}@{:d}".format(suffix, limit)
                key_num_pos = "num_pos{}@{:d}".format(suffix, limit)
                result[key_num_pos] = stats["num_pos"]
                result[key_ar] = stats["ar"].item()
                key_recalls = "Recalls{}@{:d}".format(suffix, limit)
                # result[key_recalls] = stats["recalls"]
                print(key_recalls, stats["recalls"])
                print(key_ar, "ar={:.4f}".format(result[key_ar]))
                print(key_num_pos, "num_pos={:d}".format(result[key_num_pos]))
                if limit != 1000 and dataset.relation_on:
                    # if True:
                    # relation @ 1000 (all and large) takes about 2 hs to compute
                    # relation pair evaluation
                    logger.info("Evaluating relation proposals@{:d}".format(limit))
                    stats = evaluate_box_proposals_for_relation(
                        predictions, dataset, area=area, limit=limit
                    )
                    key_ar = "AR{}@{:d}_for_relation".format(suffix, limit)
                    key_num_pos = "num_pos{}@{:d}_for_relation".format(suffix, limit)
                    result[key_num_pos] = stats["num_pos"]
                    result[key_ar] = stats["ar"].item()
                    # key_recalls = "Recalls{}@{:d}_for_relation".format(suffix, limit)
                    # result[key_recalls] = stats["recalls"]
                    print(key_ar, "ar={:.4f}".format(result[key_ar]))
                    print(key_num_pos, "num_pos={:d}".format(result[key_num_pos]))
        logger.info(result)
        # check_expected_results(result, expected_results, expected_results_sigma_tol)
        if output_folder and save_predictions:
            if box_only == 1:
                torch.save(result, os.path.join(output_folder, "rpn_proposals.pth"))
            elif box_only == 2:
                torch.save(result, os.path.join(output_folder, "box_proposals.pth"))
            else:
                raise ValueError("box_only can be either 0/1/2, but get {0}".format(box_only))
        return VGResults('box_proposal', result["AR@100"]), {"box_proposal": result}

    pred_boxlists = []
    gt_boxlists = []
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        if len(prediction) == 0:
            continue
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        pred_boxlists.append(prediction)

        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_boxlists.append(gt_boxlist)
    if eval_attributes:
        classes = dataset.attributes
    else:
        classes = dataset.classes
    result = eval_detection_voc(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        classes=classes,
        iou_thresh=0.5,
        eval_attributes=eval_attributes,
        use_07_metric=False,
    )
    result_str = "mAP: {:.4f}\n".format(result["map"])
    logger.info(result_str)
    for i, ap in enumerate(result["ap"]):
        # if i == 0:  # skip background
        #     continue
        # we skipped background in result['ap'], so we need to use i+1
        if eval_attributes:
            result_str += "{:<16}: {:.4f}\n".format(
                dataset.map_attribute_id_to_attribute_name(i+1), ap
            )
        else:
            result_str += "{:<16}: {:.4f}\n".format(
                dataset.map_class_id_to_class_name(i+1), ap
            )
    # return mAP and weighted mAP
    vg_result = VGResults('bbox', result["map"])
    if eval_attributes:
        if output_folder and save_predictions:
            with open(os.path.join(output_folder, "result_attr.txt"), "w") as fid:
                fid.write(result_str)
        return vg_result, {"attr": {"map": result["map"], "weighted map": result["weighted map"]}}
    else:
        if output_folder and save_predictions:
            with open(os.path.join(output_folder, "result_obj.txt"), "w") as fid:
                fid.write(result_str)
        return vg_result, {"obj": {"map": result["map"], "weighted map": result["weighted map"]}},


def eval_detection_voc(pred_boxlists, gt_boxlists, classes, iou_thresh=0.5, eval_attributes=False, use_07_metric=False):
    """Evaluate on voc dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."

    aps = []
    nposs = []
    thresh = []

    for i, classname in enumerate(classes):
        if classname == "__background__" or classname == "__no_attribute__":
            continue
        rec, prec, ap, scores, npos = calc_detection_voc_prec_rec(pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, \
                                                                  classindex=i, iou_thresh=iou_thresh,
                                                                  eval_attributes=eval_attributes,
                                                                  use_07_metric=use_07_metric)
        # Determine per class detection thresholds that maximise f score
        # if npos > 1:
        if npos > 1 and type(scores) != np.int:
            f = np.nan_to_num((prec * rec) / (prec + rec))
            thresh += [scores[np.argmax(f)]]
        else:
            thresh += [0]
        aps += [ap]
        nposs += [float(npos)]
        # print('AP for {} = {:.4f} (npos={:,})'.format(classname, ap, npos))
        # if pickle:
        #     with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
        #         cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap, 
        #             'scores': scores, 'npos':npos}, f)

    # Set thresh to mean for classes with poor results 
    thresh = np.array(thresh)
    avg_thresh = np.mean(thresh[thresh != 0])
    thresh[thresh == 0] = avg_thresh
    # if eval_attributes:
    #     filename = 'attribute_thresholds_' + self._image_set + '.txt'
    # else:
    #     filename = 'object_thresholds_' + self._image_set + '.txt'
    # path = os.path.join(output_dir, filename)       
    # with open(path, 'wt') as f:
    #     for i, cls in enumerate(classes[1:]):
    #         f.write('{:s} {:.3f}\n'.format(cls, thresh[i]))           

    weights = np.array(nposs)
    weights /= weights.sum()
    # print('Mean AP = {:.4f}'.format(np.mean(aps)))
    # print('Weighted Mean AP = {:.4f}'.format(np.average(aps, weights=weights)))
    # print('Mean Detection Threshold = {:.3f}'.format(avg_thresh))
    # print('~~~~~~~~')
    # print('Results:')
    # for ap, npos in zip(aps, nposs):
    #     print('{:.3f}\t{:.3f}'.format(ap, npos))
    # print('{:.3f}'.format(np.mean(aps)))
    # print('~~~~~~~~')
    # print('')
    # print('--------------------------------------------------------------')
    # print('Results computed with the **unofficial** PASCAL VOC Python eval code.')
    # print('--------------------------------------------------------------')

    # pdb.set_trace()
    return {"ap": aps, "map": np.mean(aps), "weighted map": np.average(aps, weights=weights)}


def calc_detection_voc_prec_rec(pred_boxlists, gt_boxlists, classindex, iou_thresh=0.5, eval_attributes=False,
                                use_07_metric=False):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    class_recs = {}
    npos = 0
    image_ids = []
    confidence = []
    BB = []
    for image_index, (gt_boxlist, pred_boxlist) in enumerate(zip(gt_boxlists, pred_boxlists)):
        pred_bbox = pred_boxlist.bbox.numpy()
        gt_bbox = gt_boxlist.bbox.numpy()
        if eval_attributes:
            gt_label = gt_boxlist.get_field("attributes").numpy()
            pred_label = pred_boxlist.get_field("attr_labels").numpy()
            pred_score = pred_boxlist.get_field("attr_scores").numpy()
        else:
            gt_label = gt_boxlist.get_field("labels").numpy()
            pred_label = pred_boxlist.get_field("labels").numpy()
            pred_score = pred_boxlist.get_field("scores").numpy()

        # get the ground truth information for this class
        if eval_attributes:
            gt_mask_l = np.array([classindex in i for i in gt_label])
        else:
            gt_mask_l = gt_label == classindex
        gt_bbox_l = gt_bbox[gt_mask_l]
        gt_difficult_l = np.zeros(gt_bbox_l.shape[0], dtype=bool)
        det = [False] * gt_bbox_l.shape[0]
        npos = npos + sum(~gt_difficult_l)
        class_recs[image_index] = {'bbox': gt_bbox_l,
                                   'difficult': gt_difficult_l,
                                   'det': det}

        # prediction output for each class
        # pdb.set_trace()
        if eval_attributes:
            pred_mask_l = np.logical_and(pred_label == classindex, np.not_equal(pred_score, 0.0)).nonzero()
            pred_bbox_l = pred_bbox[pred_mask_l[0]]
            pred_score_l = pred_score[pred_mask_l]
        else:
            pred_mask_l = pred_label == classindex
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]

        for bbox_tmp, score_tmp in zip(pred_bbox_l, pred_score_l):
            image_ids.append(image_index)
            confidence.append(float(score_tmp))
            BB.append([float(z) for z in bbox_tmp])

    if npos == 0:
        # No ground truth examples
        return 0, 0, 0, 0, npos

    if len(confidence) == 0:
        # No detection examples
        return 0, 0, 0, 0, npos

    confidence = np.array(confidence)
    BB = np.array(BB)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = -np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > iou_thresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap, sorted_scores, npos


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


# inspired from Detectron
def evaluate_box_proposals_for_relation(
        predictions, dataset, thresholds=None, area="all", limit=None
):
    """Evaluate how many relation pairs can be captured by the proposed boxes.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))

        # deal with ground truth
        gt_boxes = dataset.get_groundtruth(image_id)
        # filter out the field "relation_labels"
        gt_triplets = gt_boxes.get_field("relation_labels")
        if len(gt_triplets) == 0:
            continue
        gt_boxes = gt_boxes.copy_with_fields(['attributes', 'labels'])
        # get union bounding boxes (the box that cover both)
        gt_relations = getUnionBBox(gt_boxes[gt_triplets[:, 0]], gt_boxes[gt_triplets[:, 1]], margin=0)
        gt_relations.add_field('rel_classes', gt_triplets[:, 2])
        # focus on the range interested
        gt_relation_areas = gt_relations.area()
        valid_gt_inds = (gt_relation_areas >= area_range[0]) & (gt_relation_areas <= area_range[1])
        gt_relations = gt_relations[valid_gt_inds]

        num_pos += len(gt_relations)

        if len(gt_relations) == 0:
            continue

        # sort predictions in descending order and limit to the number we specify
        # TODO maybe remove this and make it explicit in the documentation
        _gt_overlaps = torch.zeros(len(gt_relations))
        if len(prediction) == 0:
            gt_overlaps.append(_gt_overlaps)
            continue
        if "objectness" in prediction.extra_fields:
            inds = prediction.get_field("objectness").sort(descending=True)[1]
        elif "scores" in prediction.extra_fields:
            inds = prediction.get_field("scores").sort(descending=True)[1]
        else:
            raise ValueError("Neither objectness nor scores is in the extra_fields!")
        prediction = prediction[inds]
        if limit is not None and len(prediction) > limit:
            prediction = prediction[:limit]
        # get the predicted relation pairs
        N = len(prediction)
        map_x = np.arange(N)
        map_y = np.arange(N)
        map_x_g, map_y_g = np.meshgrid(map_x, map_y)
        anchor_pairs = torch.from_numpy(np.vstack((map_y_g.ravel(), map_x_g.ravel())).transpose())
        # remove diagonal pairs
        keep = anchor_pairs[:, 0] != anchor_pairs[:, 1]
        anchor_pairs = anchor_pairs[keep]
        # get anchor_relations
        # anchor_relations = getUnionBBox(prediction[anchor_pairs[:,0]], prediction[anchor_pairs[:,1]], margin=0)
        if len(anchor_pairs) == 0:
            continue

        overlaps_sub = boxlist_iou(prediction[anchor_pairs[:, 0]], gt_boxes[gt_triplets[valid_gt_inds, 0]])
        overlaps_obj = boxlist_iou(prediction[anchor_pairs[:, 1]], gt_boxes[gt_triplets[valid_gt_inds, 1]])
        overlaps = torch.min(overlaps_sub, overlaps_obj)

        for j in range(min(len(anchor_pairs), len(gt_relations))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal pair that covers the best covered gt pair
            pair_ind = argmax_overlaps[gt_ind]
            # record the co-iou coverage of this gt pair
            _gt_overlaps[j] = overlaps[pair_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal pair and the gt pair as used
            overlaps[pair_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }
