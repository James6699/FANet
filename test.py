import torch
import numpy as np
from PIL import Image
import os

# 定义输入文件夹和输出文件夹的路径
input_folder = '/data/FANet-master/Evaluate-SOD-master/pred/FANET/GCPA'
output_folder = '/data/FANet-master/Evaluate-SOD-master/pred/FANET/GCPA-'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
        # 构建输入图像的完整路径
        input_path = os.path.join(input_folder, filename)
        
        # 构建输出图像的完整路径
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')
        
        # 打开JPEG图像并保存为PNG图像
        with Image.open(input_path) as img:
            img.save(output_path, 'PNG')
# # 创建一个形状为(3, 128, 20, 16, 16)的示例PyTorch张量
# # 你可以用你自己的数据来代替这个示例数据
# tensor = torch.rand(3, 128, 20, 16, 16)
# # 提取后三个通道的数据
# list1=[]

# for i in range (0,3):
#     list=[]
#     for j in range (0,128):
#         output = tensor[i, j, :, :,: ].numpy()
#         list.append(output)
#     list = np.array(list)
#     list1.append(list)
# list1 = np.array(list1)
# # 转换为NumPy数组
# tensor_1 = torch.from_numpy(list1)

# print('1')