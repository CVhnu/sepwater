from skimage.metrics import structural_similarity as ssim
import cv2

# 读取两张图像
original_img = cv2.imread('C:/Users/narob/Desktop/2/1.jpg', cv2.IMREAD_GRAYSCALE)  # 假设是原始图像
distorted_img = cv2.imread('C:/Users/narob/Desktop/2/2.jpg', cv2.IMREAD_GRAYSCALE)  # 假设是经过某些处理的图像
original_img = cv2.resize(original_img, (400, 400))
# 计算SSIM
ssim_index = ssim(original_img, distorted_img)

print(f'aver-SSIM: {ssim_index}')