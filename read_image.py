import numpy as np
import h5py
import tensorflow as tf
import cv2
    
    # C:\Users\Lenovo\Desktop\cat\cats\cats\datasets
def load_dataset():
    train_dataset = h5py.File('C:\\Users\\Lenovo\\Desktop\\cat\\cats\\cats\\datasets\\train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('C:\\Users\\Lenovo\\Desktop\\cat\\cats\\cats\\datasets\\test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

x1,x2,x3,x4,x5 = load_dataset()
print(x1.shape)

print(x2.shape)
print(x3.shape)
print(x4.shape)
print(x5)

for i in range(209):
    if x2[0,i] == 0:
        image = x1[i,:,:,:]
        cv2.imwrite('C:\\Users\\Lenovo\\Desktop\\cat\\cats\\cats\\datasets\\train\\non-cat\\'+str(i)+'.png',image)
    else:
        image = x1[i,:,:,:]
        cv2.imwrite('C:\\Users\\Lenovo\\Desktop\\cat\\cats\\cats\\datasets\\train\\cat\\'+str(i)+'.png',image)
print(image.shape)

# for i in range(50):
#     if x4[0,i] == 0:
#         image = x3[i,:,:,:]
#         cv2.imwrite('C:\\Users\\Lenovo\\Desktop\\cat\\cats\\cats\\datasets\\test\\non-cat\\'+str(i)+'.png',image)
#     else:
#         image = x3[i,:,:,:]
#         cv2.imwrite('C:\\Users\\Lenovo\\Desktop\\cat\\cats\\cats\\datasets\\test\\cat\\'+str(i)+'.png',image)
