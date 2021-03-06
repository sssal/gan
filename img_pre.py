import cv2
import numpy as np
from glob import glob

# 读取图片
def get_image(img_path,output_height=48,output_width=48):
    image = cv2.imread(img_path,flags=0) #得到灰度图
    image_b = cv2.resize(image,(output_height,output_width))
    return image_b.astype(np.float)

# 标准化
def stand(image):
    return np.array(image)/127.5-1.

def stand_r(image):
    return np.array((image+1)*127.5).astype(np.uint8)

# 一维
def twod2oned(image,output_height=48,output_width=48):
    return np.resize(image,[output_height*output_width])

def oned2twod(image,output_height=48,output_width=48):
    return np.resize(image,[output_height,output_width])

def pre_processing(img_path,output_height=48,output_width=48):
    img = get_image(img_path)
    img = stand(img)
    img = twod2oned(img)
    return img

def af_processing(img):
    img = oned2twod(img)
    img = stand_r(img)
    return img

# if __name__ == '__main__':
#     data_path = r'./data/faces/*.jpg'
#     data = glob(data_path)
#     data_path_1 = data[0]
#
#     img = pre_processing(data_path_1)
#     img_af = af_processing(img)
#
#     # img = get_image(data_path_1)
#     cv2.imshow('img',img_af)
#     cv2.waitKey(0)