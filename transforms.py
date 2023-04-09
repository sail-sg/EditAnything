# Code from https://github.com/LUSSeg/ImageNetSegModel/blob/main/util/transforms.py
from __future__ import division

import math
import random
import warnings
from collections import Iterable

import numpy as np
import torch
from torchvision.transforms import functional as F

try:
    from torchvision.transforms import InterpolationMode

    NEAREST = InterpolationMode.NEAREST
    BILINEAR = InterpolationMode.BILINEAR
    BICUBIC = InterpolationMode.BICUBIC
    LANCZOS = InterpolationMode.LANCZOS
    HAMMING = InterpolationMode.HAMMING
    HAMMING = InterpolationMode.HAMMING

    _pil_interpolation_to_str = {
        InterpolationMode.NEAREST: 'InterpolationMode.NEAREST',
        InterpolationMode.BILINEAR: 'InterpolationMode.BILINEAR',
        InterpolationMode.BICUBIC: 'InterpolationMode.BICUBIC',
        InterpolationMode.LANCZOS: 'InterpolationMode.LANCZOS',
        InterpolationMode.HAMMING: 'InterpolationMode.HAMMING',
        InterpolationMode.BOX: 'InterpolationMode.BOX',
    }

except:
    from PIL import Image

    NEAREST = Image.NEAREST
    BILINEAR = Image.BILINEAR
    BICUBIC = Image.BICUBIC
    LANCZOS = Image.LANCZOS
    HAMMING = Image.HAMMING
    HAMMING = Image.HAMMING

    _pil_interpolation_to_str = {
        Image.NEAREST: 'PIL.Image.NEAREST',
        Image.BILINEAR: 'PIL.Image.BILINEAR',
        Image.BICUBIC: 'PIL.Image.BICUBIC',
        Image.LANCZOS: 'PIL.Image.LANCZOS',
        Image.HAMMING: 'PIL.Image.HAMMING',
        Image.BOX: 'PIL.Image.BOX',
    }

def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError('Unexpected type {}'.format(type(img)))


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects):
            list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, gt):
        for t in self.transforms:
            if 'RandomResizedCrop' in t.__class__.__name__:
                img, gt = t(img, gt)
            elif 'Flip' in t.__class__.__name__:
                img, gt = t(img, gt)
            elif 'ToTensor' in t.__class__.__name__:
                img, gt = t(img, gt)
            else:
                img = t(img)
        gt = gt.float()

        return img, gt

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, gt):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(gt)
        return img, gt

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size
    and a random aspect ratio (default: of 3/4 to 4/3) of the original
    aspect ratio is made. This crop is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation=BILINEAR):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn('range should be of kind (min, max)')

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple):
                range of size of the origin size cropped
            ratio (tuple):
            range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img, gt):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(
            img, i, j, h, w, self.size, self.interpolation), \
            F.resized_crop(
                gt, i, j, h, w, self.size, NEAREST)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(
            tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(
            tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string

class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of
    shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the
    modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """
    def __call__(self, pic, gt):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), torch.from_numpy(np.array(gt))

    def __repr__(self):
        return self.__class__.__name__ + '()'