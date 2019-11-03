import os
import tensorflow as tf 
import h5py
import cv2
import numpy as np
from nets.vgg_trans import vgg16_head, bbox_reg, get_variables_in_checkpoint_file, get_variables_to_restore

# def load_dataset():
#     train_dataset = h5py.File('C:\\Users\\Lenovo\\Desktop\\cat\\cats\\cats\\datasets\\train_catvnoncat.h5', "r")
#     train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
#     train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

#     test_dataset = h5py.File('C:\\Users\\Lenovo\\Desktop\\cat\\cats\\cats\\datasets\\test_catvnoncat.h5', "r")
#     test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
#     test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

#     classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
#     train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
#     test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
#     return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# _,_,test_images,test_labels,_ = load_dataset()
# print(test_images.shape,test_labels.shape)
# test_images = test_images/255

image = cv2.imread('C:\\Users\\Lenovo\\Desktop\\cat\\cats\\cats\\datasets\\test\\non-cat\\14.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = image[np.newaxis,:,:,:]
std_image = tf.image.per_image_standardization(image)

image_placeholder = tf.placeholder(tf.float32,[None,64,64,3],name='image_placeholder')
label_placeholder = tf.placeholder(tf.int32,[1,50])

heat_map = vgg16_head(image_placeholder,False)
out = bbox_reg(heat_map,False)
predict = tf.arg_max(out,1)
# correct = tf.equal(predict,test_labels)
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess,'save_4/model.ckpt')
    test_images = sess.run(std_image)
    test_images = test_images[np.newaxis,:,:,:]
    output = sess.run(out,feed_dict = {image_placeholder:test_images})
    print(output)

    output_1 = sess.run(predict,feed_dict = {image_placeholder:test_images})
    print(output_1)
    # output_2 = sess.run(correct,feed_dict = {image_placeholder:test_images})
    # print(output_2)

