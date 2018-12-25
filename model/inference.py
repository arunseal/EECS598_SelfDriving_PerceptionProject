import tensorflow as tf
import pandas as pd
import numpy as np
import os
from alexnet import AlexNet
from vgg_16 import Vgg16
import glob
import cv2
import time

def parse_filename(filename):
    base,image = os.path.split(filename)
    _ , guid = os.path.split(base)
    return str(guid) + '/' + str(image[:4])

def read_image(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img.astype(np.float32), (224,224))
    img = img.reshape((1,224,224,3))
    return img


directory = os.path.join(os.getcwd(),'deploy','test','**','*.jpg')
weights_path = os.path.join(os.getcwd(),'weights','vgg16.npy')
list_of_filenames = glob.glob(directory)
print(len(list_of_filenames))
labels = []
names = []
images = []
num_classes = 3
x = tf.placeholder(tf.float32, [1, 224, 224, 3])
keep_prob = tf.placeholder(tf.float32)
model = Vgg16(x,num_classes,weights_path)
score = model.fc8
softmax = tf.nn.softmax(score)
saver = tf.train.Saver()

start = time.time()
print('Reading images')
for filename in list_of_filenames:
    # print('Now predicting for {}'.format(filename))
    names.append(parse_filename(filename))
    images.append(read_image(filename))
print('Done reading images')
print('Total test images are: ', len(images))
print('Time taken: ', time.time()-start , 'seconds')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.import_meta_graph('checkpoints/model_epoch1.ckpt.meta')
    saver.restore(sess, "checkpoints_vgg/model_epoch20.ckpt")
    print('Starting inference.. ')
    start_time = time.time()
    for img in images:
        # print('Now predicting for {}'.format(filename))
        # names.append(parse_filename(filename))
        # img_string = tf.read_file(filename)
        # img_decoded = tf.image.decode_png(img_string, channels=3)
        # img = tf.image.resize_images(img_decoded, [227, 227])
        # img = img.reshape((1,227,227,3))
        probs = sess.run(softmax, feed_dict={x: img, keep_prob: 1})
        label = np.argmax(probs)
        labels.append(label)
    print('Inference done.. ', time.time() - start_time)

df = pd.DataFrame({'guid/image':names,'label':labels})
df.to_csv('test_predictions_vgg_20.csv',index=False)
