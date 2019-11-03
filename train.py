import os

import numpy as np

import tensorflow as tf

from Prework import get_file, get_batch

from nets.vgg_trans import vgg16_head, bbox_reg, get_variables_in_checkpoint_file, get_variables_to_restore

# a = np.random.random((16,64,64,3))
# b = np.ones((16))

N_CLASSES = 2
IMG_W = 64
IMG_H = 64
BATCH_SIZE = 8
CAPACITY = 200
MAX_STEP = 10000
learning_rate = 0.001

train_dir = 'C:\\Users\\Lenovo\\Desktop\\cat\\cats\\cats\\datasets\\train_more'
logs_train_dir = './CK+_part'  #save path
train, train_label = get_file(train_dir)
train_batch, train_label_batch = get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

test_dir = 'C:\\Users\\Lenovo\\Desktop\\cat\\cats\\cats\\datasets\\test'
test, test_label = get_file(train_dir)
test_batch, test_label_batch = get_batch(test, test_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

image_placeholder = tf.placeholder(tf.float32,[None,64,64,3],name='image_placeholder')
label_placeholder = tf.placeholder(tf.int32,[BATCH_SIZE])

heat_map = vgg16_head(image_placeholder,True)
out = bbox_reg(heat_map,True)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = label_placeholder, logits = out)
loss = tf.reduce_sum(cross_entropy)

decay_rate = 0.5
decay_steps = 100 
global_step = tf.Variable(0)  
learning_rate = tf.train.exponential_decay(0.001, global_step, decay_steps, decay_rate, staircase=True)
# optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cross_entropy,global_step=global_step)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)


saver = tf.train.Saver()

_scope = 'vgg_16'
variables = tf.global_variables()
print('variabels',variables)
pretrained_model = os.path.join('C:\\Users\\Lenovo\\Desktop\\cat\\cats\\cats\\nets','vgg_16.ckpt')
var_keep_dic = get_variables_in_checkpoint_file(pretrained_model)

for key in var_keep_dic:
    print("tensor_name: ", key)

variables_to_restore = get_variables_to_restore(variables, var_keep_dic)
print('111')
print(variables_to_restore)

with tf.Session() as sess:

    sess.run(
            (tf.global_variables_initializer(),
            tf.local_variables_initializer()
        ))
    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, pretrained_model)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess,coord)

    training_rounds = 5000

    for i in range(training_rounds):
        train_batch_x, train_label_batch_x = sess.run([train_batch, train_label_batch])
        # print(train_batch_x[0,:,:,:])
        # print(train_label_batch_x)
        sess.run(optimizer,feed_dict={image_placeholder:train_batch_x,label_placeholder:train_label_batch_x})
        if i%100 ==0:
            test_batch_x,test_label_batch_x = sess.run([test_batch, test_label_batch])
            c = sess.run(loss,feed_dict={image_placeholder:train_batch_x,label_placeholder:train_label_batch_x})
            d = sess.run(loss,feed_dict={image_placeholder:test_batch_x,label_placeholder:test_label_batch_x})
            print(i,c,d)

    saver.save(sess,'save_4/model.ckpt')
    coord.request_stop()
    coord.join(threads)
        
