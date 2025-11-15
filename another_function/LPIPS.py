import lpips
from PIL import Image
import torchvision.transforms as transforms

# 初始化LPIPS模型，可以选择不同的预训练网络（如'alex'、'vgg'）
lpips_model = lpips.LPIPS(net='alex')

# 加载两张图像
original_img = Image.open('C:/Users/narob/Desktop/2/1.jpg').convert('RGB')
distorted_img = Image.open('C:/Users/narob/Desktop/2/2.jpg').convert('RGB')

# 预处理：将图像转换为PyTorch张量，并进行归一化
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
])
image1_tensor = preprocess(original_img).unsqueeze(0)  # 添加批次维度
image2_tensor = preprocess(distorted_img).unsqueeze(0)

# 计算LPIPS
lpips_value = lpips_model(image1_tensor, image2_tensor).item()

print(f'aver-LPIPS: {lpips_value}')