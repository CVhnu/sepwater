from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import numpy as np
from PIL import Image
from Noise_Layer import corrupt, corruption_dict


class IdentityAugment(object):
    def __call__(self, x):
        return x

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class random_select(object):
    # transform id 5 (motion blur) and 7 (snow) requires WandImage which is not fork-safe,
    # while id 4 (glass blur) and 6 (zoom blur) are super slow thus we move it to validation (unseen)

    methods = {
        'train': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        'val': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        'test': np.array([])
    }

    method_names = list(corruption_dict.keys())

    def __init__(self, min_severity=1, max_severity=5, phase='all', p=1.0, n=19):
        assert phase in ['train', 'val', 'test', 'all'], \
            ValueError(f'{phase} not recognised. Must be one of [train, val, all]')

        if phase == 'all':
            self.corrupt_ids = np.concatenate(list(self.methods.values()))
        else:
            self.corrupt_ids = self.methods[phase]

        self.corrupt_ids = self.corrupt_ids[:n]  # first n tforms
        self.phase = phase
        self.severity = np.arange(min_severity, max_severity + 1)
        self.p = p  # probability to apply a transformation

    def _apply_corruption(self, x, severity, corrupt_id):
        """Internal helper to apply the corruption logic used in multiple methods."""
        org_size = x.size
        # Resize to 224x224, convert to array, flip channels (RGB->BGR)
        x = np.asarray(x.convert('RGB').resize((224, 224), Image.BILINEAR))[:, :, ::-1]

        # Apply corruption
        x = corrupt(x, severity, corruption_number=corrupt_id)

        # Flip channels back (BGR->RGB) and convert to Image
        x = Image.fromarray(x[:, :, ::-1])

        # Resize back to original size if needed
        if x.size != org_size:
            x = x.resize(org_size, Image.BILINEAR)
        return x

    def __call__(self, x, corrupt_id=None, corrupt_strength=None):
        # input: x PIL image
        if corrupt_id is None:
            if len(self.corrupt_ids) == 0:  # do nothing
                return x
            corrupt_id = np.random.choice(self.corrupt_ids)
        else:
            assert corrupt_id in range(19)

        if corrupt_strength is None:
            severity = np.random.choice(self.severity)
        else:
            severity = corrupt_strength

        assert severity in self.severity, f'Error! Corrupt strength {severity} isnt supported.'

        if np.random.rand() < self.p:
            x = self._apply_corruption(x, severity, corrupt_id)

        return x

    def transform_with_fixed_severity(self, x, severity, corrupt_id=None):
        if corrupt_id is None:
            corrupt_id = np.random.choice(self.corrupt_ids)
        else:
            assert corrupt_id in self.corrupt_ids

        assert 0 < severity < 6

        return self._apply_corruption(x, severity, corrupt_id)

    def __repr__(self):
        s = f'(severity={self.severity}, phase={self.phase}, p={self.p}, ids={self.corrupt_ids})'
        return self.__class__.__name__ + s


class NoiseResidual(object):
    def __init__(self, k=16):
        self.k = k

    def __call__(self, x):
        h, w = x.height, x.width
        # Downscale and upscale to create a blurry version
        x1 = x.resize((w // self.k, h // self.k), Image.BILINEAR).resize((w, h), Image.BILINEAR)

        # Calculate absolute difference
        x1 = np.abs(np.array(x).astype(np.float32) - np.array(x1).astype(np.float32))

        # Normalize
        x1 = (x1 - x1.min()) / (x1.max() - x1.min() + np.finfo(np.float32).eps)
        x1 = Image.fromarray((x1 * 255).astype(np.uint8))
        return x1

    def __repr__(self):
        s = f'(k={self.k})'  # Fixed missing closing parenthesis in original string logic if needed, but kept logic mostly same
        return self.__class__.__name__ + s


def get_transforms(img_mean=[0.5, 0.5, 0.5], img_std=[0.5, 0.5, 0.5], rsize=256, csize=224,
                   pertubation=True, dct=False, residual=False, max_c=19):
    from torchvision import transforms

    prep = transforms.Compose([
        transforms.Resize(rsize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(csize)
    ])

    if pertubation:
        pertubation_train = random_select(max_severity=5, phase='train', p=0.95, n=max_c)
        pertubation_val = random_select(max_severity=5, phase='train', p=1.0, n=max_c)
        pertubation_test = random_select(max_severity=5, phase='val', p=1.0, n=max_c)
    else:
        pertubation_train = pertubation_val = pertubation_test = IdentityAugment()

    norm_steps = []
    if dct:
        from .image_tools import DCT
        norm_steps.append(DCT())

    norm_steps.append(transforms.ToTensor())
    norm_steps.append(transforms.Normalize(mean=img_mean, std=img_std))

    if residual:
        norm_steps.insert(0, NoiseResidual())

    norm_transform = transforms.Compose(norm_steps)

    preprocess = {
        'train': [prep, pertubation_train, norm_transform],
        'val': [prep, pertubation_val, norm_transform],
        'test_unseen': [prep, pertubation_test, norm_transform],
        'clean': transforms.Compose([transforms.Resize(csize), norm_transform])
    }

    return preprocess
