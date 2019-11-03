
import numpy as np

import cv2
import os

file_dir = 'C:\\Users\\Lenovo\\Desktop\\cat\\cats\\cats\\datasets\\train_more'
for file in os.listdir(file_dir + '/non-cat'):
    image_path = os.path.join(file_dir,'non-cat',file)
    print(image_path)
    src = cv2.imread(image_path)
    result3 = cv2.flip(src,1)
    cv2.imwrite(os.path.join(file_dir,'non-cat_flip',file),src)
    cv2.imwrite(os.path.join(file_dir,'non-cat_flip','f_'+file),result3)
 
