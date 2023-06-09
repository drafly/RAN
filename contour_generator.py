import os
import cv2
import shutil
from PIL import Image
import numpy as np
import fnmatch
import pandas as pd

# 产生contour
def contour(data_read, data_write):
    print(data_read)
    for name in os.listdir(data_read):
        print(name)
        mask = cv2.imread(data_read + name,0)
        mask_blur = 255 - mask

        body_clear = cv2.blur(mask, ksize=(5,5))
        body_clear = cv2.distanceTransform(body_clear, distanceType=cv2.DIST_L2, maskSize=5)
        body_clear = body_clear**0.5

        tmp  = body_clear[np.where(body_clear>0)]
        if len(tmp)!=0:
            body_clear[np.where(body_clear>0)] = np.floor(tmp/np.max(tmp)*255)

        body_blur = cv2.blur(mask_blur, ksize=(5, 5))
        body_blur = cv2.distanceTransform(body_blur, distanceType=cv2.DIST_L2, maskSize=5)
        body_blur = body_blur ** 0.5

        tmp = body_blur[np.where(body_blur > 0)]
        if len(tmp) != 0:
            body_blur[np.where(body_blur > 0)] = np.floor(tmp / np.max(tmp) * 255)

        temp_1 = body_clear
        temp_1[temp_1>0.6*255] = 0
        temp_1[temp_1 != 0] = 255
        temp_2 = body_blur
        temp_2[temp_2>0.6*255] = 0
        temp_2[temp_2 !=0 ] = 255
        contour = temp_1 + temp_2
        contour[contour != 0] = 255
        if not os.path.exists(data_write + 'contour/'):
            os.makedirs(data_write + 'contour/')
        cv2.imwrite(data_write + 'contour/' + name, contour)



if __name__ == '__main__':
    # 设置图片路径
    data_read = "./Test/"
    data_write = "./Test/"

    contour(data_read, data_write)







