#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gradcam visualisation for each GAN class
@author: Tu Bui @surrey.ac.uk
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import inspect
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision
from torch.autograd import Function
import torch.nn.functional as F


def show_cam_on_image(img, cam, cmap='jet'):
    """
    Args:
    img     PIL image (H,W,3)
    cam     heatmap (H, W), range [0,1]
    Returns:
            PIL image with heatmap applied.
    """
    cm = plt.get_cmap(cmap)
    cam = cm(cam)[...,:3]  # RGB [0,1]
    cam = np.array(img, dtype=np.float32)/255. + cam 
    cam /= cam.max()
    cam = np.uint8(cam*255)
    return Image.fromarray(cam)


class HookedModel(object):
    def __init__(self, model, feature_layer_name):
        self.model = model 
        self.feature_trees = feature_layer_name.split('.')

    def __call__(self, x):
        x = feedforward(x, self.model, self.feature_trees)
        return x 


def feedforward(x, module, layer_names):
    for name, submodule in module._modules.items():
        # print(f'Forwarding {name} ...')
        if name == layer_names[0]:
            if len(layer_names) == 1:  # leaf node reached
                # print(f'    Hook {name}')
                x = submodule(x)
                x.register_hook(save_gradients)
                save_features(x)
            else:
                # print(f'  Stepping into {name}:')
                x = feedforward(x, submodule, layer_names[1:])
        else:
            x = submodule(x)
            if name == 'avgpool':  # specific for resnet50
                x = x.view(x.size(0), -1)
    return x


basket = dict(grads=[], feature_maps=[])  # global variable to hold the gradients and output features of the layers of interest

def empty_basket():
    basket = dict(grads=[], feature_maps=[])

def save_gradients(grad):
    basket['grads'].append(grad)

def save_features(feat):
    basket['feature_maps'].append(feat)


class GradCam(object):
    def __init__(self, model, feature_layer_name, use_cuda=True):
        self.model = model 
        self.hooked_model = HookedModel(model, feature_layer_name)
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.model.eval()

    def __call__(self, x, target, act=None):
        empty_basket()
        target = torch.as_tensor(target, dtype=torch.float)
        if self.cuda:
            x = x.cuda()
            target = target.cuda()
        z = self.hooked_model(x)
        if act is not None:
            z = act(z)
        criteria = F.cosine_similarity(z, target)
        self.model.zero_grad()
        criteria.backward(retain_graph=True)
        gradients = [grad.cpu().data.numpy() for grad in basket['grads'][::-1]]  # gradients appear in reversed order
        feature_maps = [feat.cpu().data.numpy() for feat in basket['feature_maps']]
        cams = []
        for feat, grad in zip(feature_maps, gradients):
            # feat and grad have shape (1, C, H, W)
            weight = np.mean(grad, axis=(2,3), keepdims=True)[0]  # (C,1,1)
            cam = np.sum(weight * feat[0], axis=0)  # (H,w)
            cam = cv2.resize(cam, x.shape[2:])
            cam = cam - np.min(cam)
            cam = cam / (np.max(cam) + np.finfo(np.float32).eps)
            cams.append(cam)
        cams = np.array(cams).mean(axis=0)  # (H,W)
        return cams


def gradcam_demo():
    from torchvision import transforms
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()
    gradcam = GradCam(model, 'layer4.2', True)
    tform = [
                transforms.Resize((224, 224)),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
    preprocess = transforms.Compose(tform)
    im0 = Image.open('/mnt/fast/nobackup/users/tb0035/projects/diffsteg/ControlNet/examples/catdog.jpg').convert('RGB')
    im = preprocess(im0).unsqueeze(0)
    target = np.zeros((1,1000), dtype=np.float32)
    target[0, 285] = 1  # cat
    cam = gradcam(im, target)

    im0 = tform[0](im0)
    out = show_cam_on_image(im0, cam)
    out.save('test.jpg')
    print('done')


def make_target_vector(nclass, target_class_id):
    out = np.zeros((1, nclass), dtype=np.float32)
    out[0, target_class_id] = 1
    return out 



if __name__ == '__main__':
    gradcam_demo()