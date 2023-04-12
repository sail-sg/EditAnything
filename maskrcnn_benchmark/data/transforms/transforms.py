# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import random
import numpy as np
import math
import torch
import torchvision
from torchvision.transforms import functional as F

from maskrcnn_benchmark.structures.bounding_box import BoxList

def matrix_iou(a, b, relative=False):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    if relative:
        ious = area_i / (area_b[:, np.newaxis]+1e-12)
    else:
        ious = area_i / (area_a[:, np.newaxis] + area_b - area_i+1e-12)
    return ious


class RACompose(object):
    def __init__(self, pre_transforms, rand_transforms, post_transforms, concurrent=2):
        self.preprocess = pre_transforms
        self.transforms = post_transforms
        self.rand_transforms = rand_transforms
        self.concurrent = concurrent

    def __call__(self, image, target):
        for t in self.preprocess:
            image, target = t(image, target)
        for t in random.choices(self.rand_transforms, k=self.concurrent):
            image = np.array(image)
            image, target = t(image, target)
        for t in self.transforms:
            image, target = t(image, target)

        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.preprocess:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\nRandom select {0} from: (".format(self.concurrent)
        for t in self.rand_transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += ")\nThen, apply:"
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        if target is None:
            return image
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size, restrict=False):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.restrict = restrict

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if self.restrict:
            return (size, max_size)
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        if isinstance(image, np.ndarray):
            image_size = self.get_size(image.shape[:2])
            image = cv2.resize(image, image_size)
            new_size = image_size
        else:
            image = F.resize(image, self.get_size(image.size))
            new_size = image.size
        if target is not None:
            target = target.resize(new_size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            if isinstance(image, np.ndarray):
                image = np.fliplr(image)
            else:
                image = F.hflip(image)
            if target is not None:
                target = target.transpose(0)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            if isinstance(image, np.ndarray):
                image = np.flipud(image)
            else:
                image = F.vflip(image)
            target = target.transpose(1)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, format='rgb'):
        self.mean = mean
        self.std = std
        self.format = format.lower()

    def __call__(self, image, target):
        if 'bgr' in self.format:
            image = image[[2, 1, 0]]
        if '255' in self.format:
            image = image * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ColorJitter(object):
    def __init__(self,
                 brightness=0.0,
                 contrast=0.0,
                 saturation=0.0,
                 hue=0.0,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class RandomCrop(object):
    def __init__(self, prob=0.5, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.prob = prob
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, target):
        if random.random() > self.prob:
            return img, target

        h, w, c = img.shape
        boxes = target.bbox.numpy()
        labels = target.get_field('labels')

        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return img, target

            min_iou = mode

            new_w = random.uniform(self.min_crop_size * w, w)
            new_h = random.uniform(self.min_crop_size * h, h)

            # h / w in [0.5, 2]
            if new_h / new_w < 0.5 or new_h / new_w > 2:
                continue

            left = random.uniform(0, w - new_w)
            top = random.uniform(0, h - new_h)

            patch = np.array([left, top, left + new_w, top + new_h])
            overlaps = matrix_iou(patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
            if overlaps.min() < min_iou:
                continue

            # center of boxes should inside the crop img
            center = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = (center[:, 0] > patch[0]) * (center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * ( center[:, 1] < patch[3])
            if not mask.any():
                continue

            boxes = boxes[mask]
            labels = labels[mask]

            # adjust boxes
            img = img[int(patch[1]):int(patch[3]), int(patch[0]):int(patch[2])]

            boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
            boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
            boxes -= np.tile(patch[:2], 2)

            new_target = BoxList(boxes, (img.shape[1], img.shape[0]), mode='xyxy')
            new_target.add_field('labels', labels)
            return img, new_target


class RandomAffine(object):
    def __init__(self, prob=0.5, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                 borderValue=(127.5, 127.5, 127.5)):
        self.prob = prob
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.borderValue = borderValue

    def __call__(self, img, targets=None):
        if random.random() > self.prob:
            return img, targets
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

        border = 0  # width of added border (optional)
        #height = max(img.shape[0], img.shape[1]) + border * 2
        height, width, _ = img.shape
        bbox = targets.bbox

        # Rotation and Scale
        R = np.eye(3)
        a = random.random() * (self.degrees[1] - self.degrees[0]) + self.degrees[0]
        # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
        s = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = (random.random() * 2 - 1) * self.translate[0] * img.shape[0] + border  # x translation (pixels)
        T[1, 2] = (random.random() * 2 - 1) * self.translate[1] * img.shape[1] + border  # y translation (pixels)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan((random.random() * (self.shear[1] - self.shear[0]) + self.shear[0]) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan((random.random() * (self.shear[1] - self.shear[0]) + self.shear[0]) * math.pi / 180)  # y shear (deg)

        M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
        imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                                  borderValue=self.borderValue)  # BGR order borderValue

        # Return warped points also
        if targets:
            n = bbox.shape[0]
            points = bbox[:, 0:4]
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            x1 = np.clip(xy[:,0], 0, width)
            y1 = np.clip(xy[:,1], 0, height)
            x2 = np.clip(xy[:,2], 0, width)
            y2 = np.clip(xy[:,3], 0, height)
            new_bbox = np.concatenate((x1, y1, x2, y2)).reshape(4, n).T
            targets.bbox = torch.as_tensor(new_bbox, dtype=torch.float32)

        return imw, targets


class RandomErasing:
    def __init__(self, prob=0.5, era_l=0.02, era_h=1/3, min_aspect=0.3,
                 mode='const', max_count=1, max_overlap=0.3, max_value=255):
        self.prob = prob
        self.era_l = era_l
        self.era_h = era_h
        self.min_aspect = min_aspect
        self.min_count = 1
        self.max_count = max_count
        self.max_overlap = max_overlap
        self.max_value = max_value
        self.mode = mode.lower()
        assert self.mode in ['const', 'rand', 'pixel'], 'invalid erase mode: %s' % self.mode

    def _get_pixels(self, patch_size):
        if self.mode == 'pixel':
            return np.random.random(patch_size)*self.max_value
        elif self.mode == 'rand':
            return np.random.random((1, 1, patch_size[-1]))*self.max_value
        else:
            return np.zeros((1, 1, patch_size[-1]))

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        ih, iw, ic = image.shape
        ia = ih * iw
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        erase_boxes = []
        for _ in range(count):
            for try_idx in range(10):
                erase_area = random.uniform(self.era_l, self.era_h) * ia / count
                aspect_ratio = math.exp(random.uniform(math.log(self.min_aspect), math.log(1/self.min_aspect)))
                eh = int(round(math.sqrt(erase_area * aspect_ratio)))
                ew = int(round(math.sqrt(erase_area / aspect_ratio)))
                if eh < ih and ew < iw:
                    x = random.randint(0, iw - ew)
                    y = random.randint(0, ih - eh)
                    image[y:y+eh, x:x+ew, :] = self._get_pixels((eh, ew, ic))
                    erase_boxes.append([x,y,x+ew,y+eh])
                break

        if target is not None and len(erase_boxes)>0:
            boxes = target.bbox.numpy()
            labels = target.get_field('labels')
            overlap = matrix_iou(np.array(erase_boxes), boxes, relative=True)
            mask = overlap.max(axis=0)<self.max_overlap
            boxes = boxes[mask]
            labels = labels[mask]
            target.bbox = torch.as_tensor(boxes, dtype=torch.float32)
            target.add_field('labels', labels)

        return image, target
