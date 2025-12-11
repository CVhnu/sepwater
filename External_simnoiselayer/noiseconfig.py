import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from torch.cuda import device
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings
import os
from pkg_resources import resource_filename
from screenshootnoiselayer import perspective, MoireGen, Light_Distortion, Moire_Distortion, _compute_translation_matrix, _compute_tensor_center, _compute_rotation_matrix
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import kornia
import torch.nn as nn
from torch.cuda.amp import autocast
import time
from PIL import Image
from PIMoG_noise_layer import ScreenShooting
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from moire_2 import apply_finer_moire_effect_2, apply_finer_moire_effect_1
from moire_2 import apply_warm_tone
import numpy as np
from PIL import Image
from kornia.geometry.transform import warp_affine
from kornia.geometry.transform import get_rotation_matrix2d, warp_affine
import torch.nn.functional as F


warnings.simplefilter("ignore", UserWarning)
CIFAR_WH = 32



def _compute_translation_matrix_single(dx: float, dy: float, device='cuda'):
    M = torch.tensor([[[1, 0, dx],
                       [0, 1, dy]]], device=device, dtype=torch.float32)
    return M
def get_rnd_brightness_torch(rnd_bri, rnd_hue, batch_size):
    rnd_hue = torch.FloatTensor(batch_size, 3, 1, 1).uniform_(-rnd_hue, rnd_hue)
    rnd_brightness = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(-rnd_bri, rnd_bri)
    return rnd_hue, rnd_brightness


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)



wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h, w = img.shape[:2]
    # ceil crop height(= crop width)
    ch_h = int(np.ceil(h / float(zoom_factor)))
    ch_w = int(np.ceil(w / float(zoom_factor)))
    top = (h - ch_h) // 2
    left = (w - ch_w) // 2
    img = scizoom(img[top:top + ch_h, left:left + ch_w], (zoom_factor, zoom_factor, 1), order=1)
    trim_top = (img.shape[0] - h) // 2
    trim_left = (img.shape[1] - w) // 2

    return img[trim_top:trim_top + h, trim_left:trim_left + w]

#################################################################################################

def moire(x, severity=1):
    screen_shooting = ScreenShooting()
    noised_image = screen_shooting(x)
    if isinstance(noised_image, torch.Tensor):
        noised_image = torch.clamp(noised_image, 0.0, 1.0)
        noised_image = T.ToPILImage()(noised_image.squeeze(0))

    elif isinstance(noised_image, np.ndarray):
        noised_image = np.clip(noised_image, 0, 255).astype(np.uint8)
        noised_image = Image.fromarray(noised_image)

    return noised_image


def brightness(x, severity=1):
    if x.width > CIFAR_WH:
        c = [.1, .2, .3, .4, .5][severity - 1]
    else:
        c = [.05, .1, .15, .2, .3][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    if x.width > CIFAR_WH:
        c = [25, 18, 15, 10, 7][severity - 1]
    else:
        c = [80, 65, 58, 50, 40][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)

    return x



def translate(x, severity=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ct = random.uniform(0, 10)
    d = ct * severity

    image = x.convert("RGB")
    to_tensor = T.ToTensor()
    image_tensor = to_tensor(image).unsqueeze(0).to(device)

    dx = random.uniform(-d, d)
    dy = random.uniform(-d, d)

    matrix = _compute_translation_matrix_single(dx, dy, device)
    _, C, H, W = image_tensor.shape
    warped = warp_affine(image_tensor, matrix, dsize=(H, W), padding_mode='zeros')

    warped = warped.squeeze(0)
    np_img = warped.detach().cpu().numpy().transpose(1, 2, 0)
    np_img = np.clip(np_img, 0, 1) * 255
    return np_img.astype(np.uint8)


def rotate(x, severity=1):
    cr = random.uniform(0, 10)
    max_deg = cr * severity
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image = x.convert("RGB")
    img_t = T.ToTensor()(image).unsqueeze(0).to(device)
    _, C, H, W = img_t.shape

    angle = random.uniform(-max_deg, max_deg)
    angle_t = torch.tensor([angle], device=device, dtype=torch.float32)
    center = torch.tensor([[W/2-1, H/2-1]], device=device, dtype=torch.float32)
    scale = torch.tensor([[1.0, 1.0]], device=device, dtype=torch.float32)

    matrix = get_rotation_matrix2d(center, angle_t, scale)
    warped = warp_affine(img_t, matrix, dsize=(H, W), padding_mode='zeros')

    warped = warped.squeeze(0)
    np_img = warped.permute(1, 2, 0).cpu().numpy()
    np_img = np.clip(np_img, 0, 1) * 255
    return np_img.astype(np.uint8)


def perspective_noise(x, severity=1):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(np.array(x)).permute(2, 0, 1).float() / 255.0

    assert x.ndim == 3
    assert 1 <= severity <= 5

    d_values = [8, 16, 32, 48, 64]
    d = d_values[severity - 1]

    x = x.unsqueeze(0)
    c, _, h, w = x.shape
    device = x.device

    points_src = torch.tensor([[
        [0., 0.], [w - 1., 0.], [w - 1., h - 1.], [0., h - 1.]
    ]], device=device).repeat(c, 1, 1)

    points_dst = torch.zeros_like(points_src)

    for i in range(c):
        tl_x = random.uniform(-d, d)
        tl_y = random.uniform(-d, d)
        bl_x = random.uniform(-d, d)
        bl_y = random.uniform(-d, d)
        tr_x = random.uniform(-d, d)
        tr_y = random.uniform(-d, d)
        br_x = random.uniform(-d, d)
        br_y = random.uniform(-d, d)

        points_dst[i] = torch.tensor([
            [tl_x, tl_y],
            [tr_x + w - 1, tr_y],
            [br_x + w - 1, br_y + h - 1],
            [bl_x, bl_y + h - 1],
        ], device=device)

    M = kornia.geometry.get_perspective_transform(points_src, points_dst)
    warped = kornia.geometry.transform.warp_perspective(x.float(), M, dsize=(h, w))

    warped = warped.squeeze(0)  # (C, H, W)
    np_img = warped.permute(1, 2, 0).cpu().numpy()
    np_img = np.clip(np_img, 0, 1) * 255
    return np_img.astype(np.uint8)

def gaussian_noise(x, severity=1):
    if isinstance(x, np.ndarray):
        img = x.astype(np.float32)
    else:
        img = np.array(x).astype(np.float32)

    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    img = np.clip(img / 255.0 + np.random.normal(size=img.shape, scale=c), 0, 1) * 255
    return img.astype(np.uint8)

def light_distortion(x, severity=1):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(np.array(x)).permute(2, 0, 1).float() / 255.0  # (C,H,W)

    embed_image = x.unsqueeze(0)  # (1, C, H, W)
    B, C, H, W = embed_image.shape
    mask = np.ones((B, C, H, W), dtype=np.float32)
    mask_2d = np.ones((H, W), dtype=np.float32)

    a = 0.9 - severity * 0.1  # 最小暗度
    b = 1.9 + severity * 0.1  # 最大亮度

    c = random.randint(0, 1)
    if c == 0:
        direction = np.random.randint(1, 5)
        for i in range(H):
            mask_2d[i, :] = -((b - a) / (W - 1)) * (i - W) + a
        O = np.rot90(mask_2d, k=direction - 1)
        if O.shape != (H, W):
            O = O.T
        for b_idx in range(B):
            for ch in range(C):
                mask[b_idx, ch, :, :] = O
    else:
        x_center = np.random.randint(0, H)
        y_center = np.random.randint(0, W)
        max_len = max(
            np.sqrt(x_center**2 + y_center**2),
            np.sqrt((x_center - H)**2 + y_center**2),
            np.sqrt(x_center**2 + (y_center - W)**2),
            np.sqrt((x_center - H)**2 + (y_center - W)**2))
        for i in range(H):
            for j in range(W):
                val = np.sqrt((i - x_center)**2 + (j - y_center)**2) / max_len * (a - b) + b
                mask[:, :, i, j] = val

    result = embed_image * torch.tensor(mask).to(embed_image.device)
    warped = result.squeeze(0)
    np_img = warped.permute(1, 2, 0).cpu().numpy()
    np_img = np.clip(np_img, 0, 1) * 255
    return np_img.astype(np.uint8)


def eye_protection(x, severity=1):
    if not isinstance(x, np.ndarray):
        np_img = np.array(x)
    else:
        np_img = x

    blue_reduce = 0.8 - 0.1 * severity
    warm_boost = 1.05 + 0.05 * severity
    gamma = 1.0 - 0.05 * severity
    brightness = 0.9 + 0.1 * severity

    b, g, r = cv2.split(np_img.astype(np.float32))

    b *= blue_reduce
    r *= warm_boost
    b *= brightness
    g *= brightness
    r *= brightness

    img = cv2.merge([b, g, r])
    img = np.clip(img, 0, 255).astype(np.uint8)

    look_up = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    img = cv2.LUT(img, look_up)

    return img

def gradient_gray(x, severity=1):

    if not isinstance(x, np.ndarray):
        image = np.array(x)
    else:
        image = x

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    corners = {
        "tl": np.array([0, 0]),
        "tr": np.array([w - 1, 0]),
        "br": np.array([w - 1, h - 1]),
        "bl": np.array([0, h - 1])
    }
    c1, c2 = random.choice([("tl", "tr"), ("tr", "br"), ("br", "bl"), ("bl", "tl")])
    p1 = corners[c1]
    p2 = corners[c2]

    max_dist = np.linalg.norm([w, h])
    distance_map = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            d1 = np.linalg.norm(np.array([x, y]) - p1)
            d2 = np.linalg.norm(np.array([x, y]) - p2)
            distance_map[y, x] = min(d1, d2) / max_dist

    alpha = (1.0 - distance_map) ** 3 * severity
    alpha = np.clip(alpha, 0.0, 1.0)
    output = (alpha[..., None] * gray_color + (1 - alpha[..., None]) * image).astype(np.uint8)
    return output





