import cv2
import numpy as np


def generate_finer_moire_pattern(size, frequency=300, angle=45, amplitude=0.5):
    H, W = size
    x = np.arange(W)
    y = np.arange(H)
    xx, yy = np.meshgrid(x, y)
    angle_rad = np.deg2rad(angle)
    rotated_x = xx * np.cos(angle_rad) - yy * np.sin(angle_rad)
    sine_wave = amplitude * np.sin(2 * np.pi * rotated_x / frequency)
    r_channel = (np.sin(sine_wave) + 1) * 127.5
    g_channel = (np.sin(sine_wave + np.pi / 3) + 1) * 127.5
    b_channel = (np.sin(sine_wave + 2 * np.pi / 3) + 1) * 127.5
    moire_pattern = np.stack([r_channel, g_channel, b_channel], axis=-1).astype(np.uint8)
    return moire_pattern


def apply_finer_moire_effect_1(image):
    frequency = 400
    angle = 45
    amplitude = 0.5
    alpha = 0.1
    H, W = image.shape[:2]
    moire_pattern = generate_finer_moire_pattern((H, W), frequency, angle, amplitude)
    moire_pattern_resized = cv2.resize(moire_pattern, (W, H))
    blended_image = cv2.addWeighted(image, 1 - alpha, moire_pattern_resized, alpha, 0)
    return blended_image

def apply_finer_moire_effect_2(image_array):
    frequency = 300
    angle = 45
    amplitude = 0.5
    alpha = 0.2
    H, W = image_array.shape[:2]
    moire_pattern = generate_finer_moire_pattern((H, W), frequency, angle, amplitude)
    if image_array.shape != moire_pattern.shape:
        moire_pattern = cv2.resize(moire_pattern, (image_array.shape[1], image_array.shape[0]))
    if image_array.shape[2] != moire_pattern.shape[2]:
        if moire_pattern.ndim == 2:
            moire_pattern = cv2.cvtColor(moire_pattern, cv2.COLOR_GRAY2BGR)

    image_array = np.uint8(image_array)
    moire_pattern = np.uint8(moire_pattern)
    blended_image = cv2.addWeighted(image_array, 1 - alpha, moire_pattern, alpha, 0)
    blended_image[:, :, 1] = np.clip(blended_image[:, :, 1] * 1.1, 0, 255)
    blended_image[:, :, 2] = np.clip(blended_image[:, :, 2] * 1.05, 0, 255)
    return blended_image

from PIL import Image


def apply_warm_tone(image):

    if isinstance(image, np.ndarray) and image.ndim == 4:  # 批量数据 (batch_size, height, width, channels)
        image = image[0, 0]
    image = np.array(image)
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    r = np.clip(r * 1.1, 0, 255)
    b = np.clip(b * 0.9, 0, 255)
    warm_image = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(warm_image)