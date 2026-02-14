import lpips
import argparse
import os
import os.path as osp
import torch
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm


# 图像预处理函数
def preprocess_image(image_path):
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    
    # 定义预处理转换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 根据需要调整图像大小
        transforms.ToTensor(),           # 将图像转换为 Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 进行标准化
    ])
    
    # 应用转换
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0)  # 增加一个维度，变为 NCHW 格式


# 创建解析器对象
parser = argparse.ArgumentParser(description="A simple example of argparse.")

# 添加位置参数
parser.add_argument('input', type=str, help='Input file name')
parser.add_argument('output', type=str, help='Output file name')
# 解析命令行参数
args = parser.parse_args()

loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

imglist1 = os.listdir(args.input)
imglist2 = os.listdir(args.output)
imglist1 = list(filter(lambda x: x != '.DS_Store', imglist1))
imglist2 = list(filter(lambda x: x != '.DS_Store', imglist2))

imglist1 = sorted(imglist1)
imglist2 = sorted(imglist2)

loss_d = []
for (imname1, imname2) in tqdm(zip(imglist1, imglist2)):
    # import ipdb;ipdb.set_trace()
    assert imname1.replace('_0.jpg', '.jpg') == imname2.replace('_0.jpg', '.jpg')
    img1 = preprocess_image(osp.join(args.input, imname1))
    img2 = preprocess_image(osp.join(args.output, imname2))
    d = loss_fn_alex(img1, img2)
    loss_d.append(d.item())

print("mean lpips: ", torch.mean(torch.tensor(loss_d)))
