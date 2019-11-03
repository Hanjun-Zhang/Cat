
import numpy as np

import cv2
import os

file_dir = 'C:\\Users\\Lenovo\\Desktop\\cat\\cats\\cats\\datasets\\train_more'
for file in os.listdir(file_dir + '/non-cat_flip'):
    image_path = os.path.join(file_dir,'non-cat_flip',file)
    # image_path = 'C:\\Users\\Lenovo\\Desktop\\cat\\cats\\cats\\datasets\\train_more\\cat_flip\\102.png'
    print(image_path)
    image = cv2.imread(image_path)
    print(image)
    new_image = np.zeros(image.shape, image.dtype)
    alpha = 0.7 # Simple contrast control
    beta = 20 # Simple brightness control
        # try:
        #     alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
        #     beta = int(input('* Enter the beta value [0-100]: '))
        # except ValueError:
        #     print('Error, not a number')

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
                # print(new_image)
    cv2.imwrite(os.path.join(file_dir,'non-cat_bright','0.7_'+file),new_image)
    # cv2.imshow('Original Image', image)
    # cv2.imshow('New Image', new_image)
    # cv2.waitKey()
