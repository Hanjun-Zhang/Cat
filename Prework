
import os

import numpy as np

from PIL import Image

import tensorflow as tf

# import matplotlib.pyplot as plt

from numpy import *

 
cat = []
label_cat = []

noncat = []
label_noncat = []



 

 

def get_file(file_dir):


    for file in os.listdir(file_dir + '/cat_bright'):

        cat.append(file_dir + '/cat_bright' + '/' + file)

        label_cat.append(1)

    for file in os.listdir(file_dir  + '/non-cat_bright'):

        noncat.append(file_dir + '/non-cat_bright' + '/' + file)

        label_noncat.append(0)


    
    # print('there are .......')
    # print(len(cat),len(noncat))
    
    image_list = np.hstack((cat,noncat))

    label_list = np.hstack((label_cat,label_noncat))

    

    temp = np.array([image_list, label_list])   

    temp = temp.transpose()     

    

    np.random.shuffle(temp)     

 


    all_image_list = list(temp[:, 0])    

    all_label_list = list(temp[:, 1])    

    label_list = [int(i) for i in label_list]   

    return image_list, label_list

 


def get_batch(image, label, image_W, image_H, batch_size, capacity):


    image = tf.cast(image, tf.string)   
    label = tf.cast(label, tf.int32)


    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]

    image_contents = tf.read_file(input_queue[0])   
 

    image = tf.image.decode_png(image_contents, channels=3)

    

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)


    image = tf.image.per_image_standardization(image)


    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=16, capacity=capacity)


    label_batch = tf.reshape(label_batch, [batch_size])


    image_batch = tf.cast(image_batch, tf.float32)    #

    return image_batch, label_batch


 


# def PreWork():

#     # 对预处理的数据进行可视化，查看预处理的效果

#     IMG_W = 64

#     IMG_H = 64

#     BATCH_SIZE = 6

#     CAPACITY = 64



#     train_dir = 'C:\\Users\\Lenovo\\Desktop\\cat\\cats\\cats\\datasets\\train'



#     # image_list, label_list, val_images, val_labels = get_file(train_dir)

#     image_list, label_list = get_file(train_dir)

#     image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

#     print(label_batch.shape)



#     lists = ('cat', 'noncat')



#     with tf.Session() as sess:

#         i = 0

#         coord = tf.train.Coordinator()  
#         threads = tf.train.start_queue_runners(coord=coord)

#         try:

#             while not coord.should_stop() and i < 1:

#                 # 提取出两个batch的图片并可视化。

#                 img, label = sess.run([image_batch, label_batch])  
#                 # img = tf.cast(img, tf.uint8)


#                 for j in np.arange(BATCH_SIZE):

#                     # np.arange()函数返回一个有终点和起点的固定步长的排列

#                     print('label: %d' % label[j])

#                     plt.imshow(img[j, :, :, :])

#                     title = lists[int(label[j])]

#                     plt.title(title)

#                     plt.show()

#                 i += 1

#         except tf.errors.OutOfRangeError:

#             print('done!')

#         finally:

#             coord.request_stop()

#         coord.join(threads)



# if __name__ == '__main__':

#     PreWork()

