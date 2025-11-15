import cv2
import os
import glob

# 指定原始图片和目标图片的文件夹路径
input_folder = "C:/Users/narob/PycharmProjects/stegastamp/data/images"
output_folder = "C:/Users/narob/PycharmProjects/stegastamp/data/test"

# 创建输出文件夹如果它不存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 设置新的尺寸
new_width = 400
new_height = 400

# 获取所有图片文件的路径
image_files = glob.glob(os.path.join(input_folder, '*'))

# 遍历所有图片文件
for image_file in image_files:
    # 读取图片
    img = cv2.imread(image_file)

    # 如果图片读取成功，则进行缩放
    if img is not None:
        # 缩放图片
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 构建输出文件的路径
        output_file = os.path.join(output_folder, os.path.basename(image_file))

        # 保存缩放后的图片
        cv2.imwrite(output_file, resized_img)
    else:
        print(f"Image {image_file} cannot be read or is a non-image file.")

print("Image resizing complete.")