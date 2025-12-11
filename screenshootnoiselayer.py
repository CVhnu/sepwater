import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import kornia
import os
import torch.nn as nn
from torch.cuda.amp import autocast
import time



def _compute_translation_matrix(translation: torch.Tensor) -> torch.Tensor:
    matrix: torch.Tensor = torch.eye(3, device=translation.device, dtype=translation.dtype)
    matrix = matrix.repeat(translation.shape[0], 1, 1)

    dx, dy = torch.chunk(translation, chunks=2, dim=-1)
    matrix[..., 0, 2:3] += dx
    matrix[..., 1, 2:3] += dy
    return matrix



def _compute_tensor_center(tensor: torch.Tensor) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    center_x = float(width - 1) / 2
    center_y = float(height - 1) / 2
    return torch.tensor([center_x, center_y], device=tensor.device, dtype=tensor.dtype)


def _compute_rotation_matrix(angle: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    scale = torch.ones((angle.shape[0], 2))
    matrix = kornia.geometry.transform.get_rotation_matrix2d(center, angle, scale)
    return matrix


def translate(image, device, d=8):
    c, h, w = image.shape
    trans = torch.ones(c, 2)
    for i in range(c):
        dx = random.uniform(-d, d)
        dy = random.uniform(-d, d)
        trans[i, :] = torch.tensor([dx, dy])
    translation_matrix = _compute_translation_matrix(trans)
    matrix = translation_matrix[..., :2, :3]
    if image.ndimension() == 3:
        image = image.unsqueeze(0)
    matrix = matrix.expand(image.shape[0], -1, -1).to(device)
    data_warp = kornia.warp_affine(image, matrix, dsize=(h, w), padding_mode='border').to(device)
    if image.ndimension() == 3:
        data_warp = data_warp.squeeze(0)

    processed_image_np = data_warp.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    processed_image_np = np.clip(processed_image_np, 0, 1) * 255  # 调整范围到 [0, 255]
    return processed_image_np.astype(np.uint8)


def rotate(image, device, d=8):
    c, h, w = image.shape
    angle = torch.ones(c)
    center = torch.ones(c, 2)
    for i in range(c):
        angle[i] = random.uniform(-d, d)
        center[i, :] = torch.tensor([h / 2 - 1, w / 2 - 1])
    angle = angle.expand(image.shape[0])
    center = center.expand(image.shape[0], -1)
    rotation_matrix = _compute_rotation_matrix(angle, center)
    matrix = rotation_matrix[..., :2, :3]
    if image.ndimension() == 3:
        image = image.unsqueeze(0)
    matrix = matrix.expand(image.shape[0], -1, -1).to(device)
    data_warp = kornia.warp_affine(image, matrix, dsize=(h, w), padding_mode='border').to(device)
    if image.ndimension() == 3:
        data_warp = data_warp.squeeze(0)


    processed_image_np = data_warp.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    processed_image_np = np.clip(processed_image_np, 0, 1) * 255  # 调整范围到 [0, 255]
    return processed_image_np.astype(np.uint8)

def perspective(image, device, d=8):
    c, h, w = image.shape[0], image.shape[2], image.shape[3]
    points_src = torch.ones(c, 4, 2)
    points_dst = torch.ones(c, 4, 2)
    for i in range(c):
        points_src[i, :, :] = torch.tensor([[0., 0.], [w - 1., 0.], [w - 1., h - 1.], [0., h - 1.]])
        tl_x = random.uniform(-d, d)
        tl_y = random.uniform(-d, d)
        bl_x = random.uniform(-d, d)
        bl_y = random.uniform(-d, d)
        tr_x = random.uniform(-d, d)
        tr_y = random.uniform(-d, d)
        br_x = random.uniform(-d, d)
        br_y = random.uniform(-d, d)
        points_dst[i, :, :] = torch.tensor([[tl_x, tl_y], [tr_x + h, tr_y], [br_x + h, br_y + h], [bl_x, bl_y + h]])
    M = kornia.geometry.get_perspective_transform(points_src, points_dst).to(device)
    return kornia.geometry.transform.warp_perspective(image.float(), M, dsize=(h, w)).to(device)



def MoireGen(p_size, theta, center_x, center_y):
    z = np.zeros((p_size, p_size))
    for i in range(p_size):
        for j in range(p_size):
            z1 = 0.5 + 0.5 * math.cos(2 * math.pi * np.sqrt((i + 1 - center_x) ** 2 + (j + 1 - center_y) ** 2))
            z2 = 0.5 + 0.5 * math.cos(math.cos(theta / 180 * math.pi) * (j + 1) + math.sin(theta / 180 * math.pi) * (i + 1))
            z[i, j] = np.min([z1, z2])
    return (z + 1) / 2



def Light_Distortion(c, embed_image):
    mask = np.zeros((embed_image.shape))
    mask_2d = np.zeros((embed_image.shape[2], embed_image.shape[3]))
    a = 0.7 + np.random.rand(1) * 0.2
    b = 1.1 + np.random.rand(1) * 0.2
    if c == 0:
        direction = np.random.randint(1, 5)
        for i in range(embed_image.shape[2]):
            mask_2d[i, :] = -((b - a) / (mask.shape[2] - 1)) * (i - mask.shape[3]) + a
        O = np.rot90(mask_2d, direction - 1)
        for batch in range(embed_image.shape[0]):
            for channel in range(embed_image.shape[1]):
                mask[batch, channel, :, :] = mask_2d
    else:
        x = np.random.randint(0, mask.shape[2])
        y = np.random.randint(0, mask.shape[3])
        max_len = np.max([np.sqrt(x ** 2 + y ** 2), np.sqrt((x - 255) ** 2 + y ** 2), np.sqrt(x ** 2 + (y - 255) ** 2),
                          np.sqrt((x - 255) ** 2 + (y - 255) ** 2)])
        for i in range(mask.shape[2]):
            for j in range(mask.shape[3]):
                mask[:, :, i, j] = np.sqrt((i - x) ** 2 + (j - y) ** 2) / max_len * (a - b) + b
        O = mask

    return O.copy()



def Moire_Distortion(embed_image):
    Z = torch.zeros_like(embed_image)
    for i in range(3):
        theta = np.random.randint(0, 180)
        center_x = np.random.rand(1) * embed_image.shape[2]
        center_y = np.random.rand(1) * embed_image.shape[3]
        M = MoireGen(embed_image.shape[2], theta, center_x, center_y)
        M_resized = np.resize(M, (embed_image.shape[2], embed_image.shape[3]))
        Z[:, i, :, :] = torch.tensor(M_resized, device=embed_image.device)
    return Z



def addshootnoise_image(image_path, severity=1, device='cuda', target_size=(224, 224)):

    if isinstance(image_path, Image.Image):
        image = image_path.convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")


    image = image.resize(target_size)

    transform = T.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)


    noised_image = torch.zeros_like(image_tensor)
    noised_image = perspective(image_tensor, device, 2)


    c = np.random.randint(0, 2)
    L = Light_Distortion(c, image_tensor)
    L = torch.tensor(L, device=image_tensor.device, dtype=torch.float32)


    Z = Moire_Distortion(image_tensor) * (2 - severity * 0.5)
    noised_image = noised_image * L * 0.85 + Z * 0.15
    noised_image = noised_image + 0.001 ** 0.5 * torch.randn(noised_image.size()).to(device)


    processed_image_np = noised_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    processed_image_np = np.clip(processed_image_np, 0, 1) * 255  # 调整范围到 [0, 255]

    return processed_image_np.astype(np.uint8)


import os
from PIL import Image

def process_images_in_folder(input_folder, output_folder, severity=1, device='cuda', target_size=(224, 224)):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    processed_files = set(os.listdir(output_folder))


    image_paths = [
        os.path.join(input_folder, f) for f in os.listdir(input_folder)
        if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp')) and f not in processed_files
    ]


    for image_path in image_paths:
        processed_image = addshootnoise_image(image_path, severity, device, target_size)


        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, image_name)
        Image.fromarray(processed_image).save(output_path)

if __name__ == '__main__':
    input_folder = "/home/xxy/Desktop/test2"
    output_folder = "/home/xxy/Desktop/test3"


    process_images_in_folder(input_folder, output_folder, severity=1)


