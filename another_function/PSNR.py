import numpy as np
import cv2

def calculate_psnr(original_image, distorted_image):
    mse = np.mean((original_image - distorted_image) ** 2)
    if mse == 0:
        return float('inf')  # 如果没有误差，PSNR是无穷大
    max_pixel = 255.0  # 最大像素值，对于8位图像是255
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# 读取原始图像和失真图像

original_img = cv2.imread("C:/Users/narob/Desktop/2/1.jpg", cv2.IMREAD_GRAYSCALE)
distorted_img = cv2.imread("C:/Users/narob/Desktop/2/2.jpg", cv2.IMREAD_GRAYSCALE)
original_img = cv2.resize(original_img, (400, 400))

# 确保图像读取成功
if original_img is None or distorted_img is None:
    raise ValueError("图像文件读取失败，请检查文件路径。")

# 计算 PSNR
psnr_value = calculate_psnr(original_img, distorted_img)
print(f"aver-PSNR: {psnr_value} dB")

