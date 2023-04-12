import torch
import numpy as np

from maskrcnn_benchmark.config import cfg
from glip.maskrcnn_benchmark.data import transforms as T
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from glip.maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from glip.maskrcnn_benchmark.layers import soft_nms
from maskrcnn_benchmark.layers import nms


def im_detect_bbox_aug(model, images, device, captions=None, positive_map_label_to_token=None):
    # Collect detections computed under different transformations
    boxlists_ts = []
    for _ in range(len(images)):
        boxlists_ts.append([])

    def add_preds_t(boxlists_t):
        for i, boxlist_t in enumerate(boxlists_t):
            # Resize the boxlist as the first one
            boxlists_ts[i].append(boxlist_t.resize(images[i].size))

    # Compute detections at different scales
    if len(cfg.TEST.RANGES)==len(cfg.TEST.SCALES):
        keep_ranges = cfg.TEST.RANGES
    else:
        keep_ranges = [None for _ in cfg.TEST.SCALES]

    for scale, keep_range in zip(cfg.TEST.SCALES, keep_ranges):
        max_size = cfg.TEST.MAX_SIZE
        boxlists_scl = im_detect_bbox_scale(
            model, images, scale, max_size, device,
            captions=captions,
            positive_map_label_to_token=positive_map_label_to_token,
        )
        if keep_range is not None:
            boxlists_scl = remove_boxes(boxlists_scl, *keep_range)
        add_preds_t(boxlists_scl)

        if cfg.TEST.FLIP:
            boxlists_scl_hf = im_detect_bbox_scale(
                model, images, scale, max_size, device,
                captions=captions,
                positive_map_label_to_token=positive_map_label_to_token,
                hflip=True
            )
            if keep_range is not None:
                boxlists_scl_hf = remove_boxes(boxlists_scl_hf, *keep_range)
            add_preds_t(boxlists_scl_hf)

    # Merge boxlists detected by different bbox aug params
    boxlists = []
    for i, boxlist_ts in enumerate(boxlists_ts):
        bbox = torch.cat([boxlist_t.bbox for boxlist_t in boxlist_ts])
        scores = torch.cat([boxlist_t.get_field('scores') for boxlist_t in boxlist_ts])
        labels = torch.cat([boxlist_t.get_field('labels') for boxlist_t in boxlist_ts])
        boxlist = BoxList(bbox, boxlist_ts[0].size, boxlist_ts[0].mode)
        boxlist.add_field('scores', scores)
        boxlist.add_field('labels', labels)
        boxlists.append(boxlist)
    results = merge_result_from_multi_scales(boxlists)
    return results


def im_detect_bbox(model, images, target_scale, target_max_size, device,
                   captions=None,
                   positive_map_label_to_token=None
                   ):
    """
    Performs bbox detection on the original image.
    """
    if cfg.INPUT.FORMAT is not '':
        input_format = cfg.INPUT.FORMAT
    elif cfg.INPUT.TO_BGR255:
        input_format = 'bgr255'
    transform = T.Compose([
        T.Resize(target_scale, target_max_size),
        T.ToTensor(),
        T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, format=input_format
        )
    ])
    images = [transform(image) for image in images]
    images = to_image_list(images, cfg.DATALOADER.SIZE_DIVISIBILITY)
    if captions is None:
        return model(images.to(device))
    else:
        return model(images.to(device),
                     captions=captions,
                     positive_map=positive_map_label_to_token
                     )


def im_detect_bbox_hflip(model, images, target_scale, target_max_size, device,
                         captions=None,
                         positive_map_label_to_token=None
                         ):
    """
    Performs bbox detection on the horizontally flipped image.
    Function signature is the same as for im_detect_bbox.
    """
    if cfg.INPUT.FORMAT is not '':
        input_format = cfg.INPUT.FORMAT
    elif cfg.INPUT.TO_BGR255:
        input_format = 'bgr255'
    transform = T.Compose([
        T.Resize(target_scale, target_max_size),
        T.RandomHorizontalFlip(1.0),
        T.ToTensor(),
        T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, format=input_format
        )
    ])
    images = [transform(image) for image in images]
    images = to_image_list(images, cfg.DATALOADER.SIZE_DIVISIBILITY)
    if captions is None:
        boxlists = model(images.to(device))
    else:
        boxlists = model(images.to(device),
                         captions=captions,
                         positive_map=positive_map_label_to_token
                         )

    # Invert the detections computed on the flipped image
    boxlists_inv = [boxlist.transpose(0) for boxlist in boxlists]
    return boxlists_inv


def im_detect_bbox_scale(model, images, target_scale, target_max_size, device,
                         captions=None,
                         positive_map_label_to_token=None,
                         hflip=False):
    """
    Computes bbox detections at the given scale.
    Returns predictions in the scaled image space.
    """
    if hflip:
        boxlists_scl = im_detect_bbox_hflip(model, images, target_scale, target_max_size, device,
                                            captions=captions,
                                            positive_map_label_to_token=positive_map_label_to_token
                                            )
    else:
        boxlists_scl = im_detect_bbox(model, images, target_scale, target_max_size, device,
                                      captions=captions,
                                      positive_map_label_to_token=positive_map_label_to_token
                                      )
    return boxlists_scl


def remove_boxes(boxlist_ts, min_scale, max_scale):
    new_boxlist_ts = []
    for _, boxlist_t in enumerate(boxlist_ts):
        mode = boxlist_t.mode
        boxlist_t = boxlist_t.convert("xyxy")
        boxes = boxlist_t.bbox
        keep = []
        for j, box in enumerate(boxes):
            w = box[2] - box[0] + 1
            h = box[3] - box[1] + 1
            if (w * h > min_scale * min_scale) and (w * h < max_scale * max_scale):
                keep.append(j)
        new_boxlist_ts.append(boxlist_t[keep].convert(mode))
    return new_boxlist_ts


def merge_result_from_multi_scales(boxlists):
    num_images = len(boxlists)
    results = []
    for i in range(num_images):
        scores = boxlists[i].get_field("scores")
        labels = boxlists[i].get_field("labels")
        boxes = boxlists[i].bbox
        boxlist = boxlists[i]
        result = []
        # test on classes
        if len(cfg.TEST.SELECT_CLASSES):
            class_list = cfg.TEST.SELECT_CLASSES
        else:
            class_list = range(1, cfg.TEST.NUM_CLASSES)
        for j in class_list:
            inds = (labels == j).nonzero().view(-1)

            scores_j = scores[inds]
            boxes_j = boxes[inds, :].view(-1, 4)
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms(boxlist_for_class, cfg.TEST.TH, score_field="scores", nms_type=cfg.TEST.SPECIAL_NMS)
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field("labels", torch.full((num_labels,), j, dtype=torch.int64, device=scores.device))
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > cfg.TEST.PRE_NMS_TOP_N > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(),
                number_of_detections - cfg.TEST.PRE_NMS_TOP_N + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        results.append(result)
    return results


def boxlist_nms(boxlist, thresh, max_proposals=-1, score_field="scores", nms_type='nms'):
    if thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)

    if nms_type == 'vote':
        boxes_vote, scores_vote = bbox_vote(boxes, score, thresh)
        if len(boxes_vote) > 0:
            boxlist.bbox = boxes_vote
            boxlist.extra_fields['scores'] = scores_vote
    elif nms_type == 'soft-vote':
        boxes_vote, scores_vote = soft_bbox_vote(boxes, score, thresh)
        if len(boxes_vote) > 0:
            boxlist.bbox = boxes_vote
            boxlist.extra_fields['scores'] = scores_vote
    elif nms_type == 'soft-nms':
        keep, new_score = soft_nms(boxes.cpu(), score.cpu(), thresh, 0.95)
        if max_proposals > 0:
            keep = keep[: max_proposals]
        boxlist = boxlist[keep]
        boxlist.extra_fields['scores'] = new_score
    else:
        keep = nms(boxes, score, thresh)
        if max_proposals > 0:
            keep = keep[: max_proposals]
        boxlist = boxlist[keep]
    return boxlist.convert(mode)


def bbox_vote(boxes, scores, vote_thresh):
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy().reshape(-1, 1)
    det = np.concatenate((boxes, scores), axis=1)
    if det.shape[0] <= 1:
        return np.zeros((0, 5)), np.zeros((0, 1))
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these  det
        merge_index = np.where(o >= vote_thresh)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score
            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    boxes = torch.from_numpy(dets[:, :4]).float().cuda()
    scores = torch.from_numpy(dets[:, 4]).float().cuda()

    return boxes, scores


def soft_bbox_vote(boxes, scores, vote_thresh):
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy().reshape(-1, 1)
    det = np.concatenate((boxes, scores), axis=1)
    if det.shape[0] <= 1:
        return np.zeros((0, 5)), np.zeros((0, 1))
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these  det
        merge_index = np.where(o >= vote_thresh)[0]
        det_accu = det[merge_index, :]
        det_accu_iou = o[merge_index]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            soft_det_accu = det_accu.copy()
            soft_det_accu[:, 4] = soft_det_accu[:, 4] * (1 - det_accu_iou)
            soft_index = np.where(soft_det_accu[:, 4] >= cfg.MODEL.RETINANET.INFERENCE_TH)[0]
            soft_det_accu = soft_det_accu[soft_index, :]

            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score

            if soft_det_accu.shape[0] > 0:
                det_accu_sum = np.row_stack((det_accu_sum, soft_det_accu))

            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    order = dets[:, 4].ravel().argsort()[::-1]
    dets = dets[order, :]

    boxes = torch.from_numpy(dets[:, :4]).float().cuda()
    scores = torch.from_numpy(dets[:, 4]).float().cuda()

    return boxes, scores