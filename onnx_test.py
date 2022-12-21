import time

import onnx

from onnxmodel import ONNXModel
from PIL import Image
import numpy as np
# from test_image import WriteDepthOnnx
from torchvision import transforms

model_time_start = time.time()
net = ONNXModel("StereoNet_400*640.onnx")
model_time_end = time.time()
print("load_model time :", model_time_end - model_time_start)

limg_ori = Image.open("images1/cam0/rbn100_0.L.png").convert('RGB')
rimg_ori = Image.open("images1/cam1/rbn100_0.R.png").convert('RGB')

gpu_time_start = time.time()
limg_tensor = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((400, 640)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])(limg_ori)
rimg_tensor = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((384, 1280)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])(rimg_ori)
limg_tensor = limg_tensor.unsqueeze(0).cuda()
rimg_tensor = rimg_tensor.unsqueeze(0).cuda()

gpu_time_end = time.time()
print("data to gpu :", gpu_time_end - gpu_time_start)

limg=limg_tensor.cpu().numpy()
rimg=rimg_tensor.cpu().numpy()

# for i in range(10):
time_start = time.time()
output  = net.forward(limg,rimg)
time_end = time.time()
print("interface time ", time_end - time_start)

dis_array = output[0][0][0]
dis_array = (dis_array - dis_array.min()) / (dis_array.max() - dis_array.min()) * 255.0
dis_array = dis_array.astype("uint8")

import cv2
showImg = cv2.resize(dis_array, (dis_array.shape[-1], dis_array.shape[0]))
showImg = cv2.applyColorMap(cv2.convertScaleAbs(showImg, 1), cv2.COLORMAP_PARULA)
cv2.imwrite("onnx_result.jpg", showImg)
