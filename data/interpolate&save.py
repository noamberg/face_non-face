import glob
import os
import cv2
import numpy as np
import pandas as pd
from utils import create_dir


def load_images_from_folder(folder):
    images = []
    for filename in glob.glob(folder):
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
    return images



# data paths
train_path = r'C:\Users\Noam\PycharmProjects\Jubaan\data\train\*\*'
test_path = r'C:\Users\Noam\PycharmProjects\Jubaan\data\test\*\*'
imgs = load_images_from_folder(test_path)
save_path = r'C:\Users\Noam\PycharmProjects\Jubaan\data\test_interpolated'
create_dir(save_path)

for idx,img in enumerate(imgs):
    resized_img = cv2.resize(img, (224, 224))
    cv2.imwrite(os.path.join(save_path, 'resized_img_{}.jpg'.format(idx)), resized_img)

print('done')
