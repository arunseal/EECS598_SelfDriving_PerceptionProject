import tensorflow as tf
import numpy as np
from vgg_16 import Vgg16
from datagenerator import ImageDataGenerator
from tensorflow.data import Iterator
from datetime import datetime
import os
import pickle


current_dir = os.getcwd()
train_file = os.path.join(current_dir,'train_v2.csv')
val_file = os.path.join(current_dir,'validation_v2.csv')
weights_path = os.path.join(current_dir,'weights','vgg16.npy')

learning_rate = 0.0001
num_epochs = 100
batch_size = 32
dropout_rate = 0.5
num_classes = 3

# filewriter_path = os.path.join(current_dir,'tensorboard')
checkpoint_path = os.path.join(current_dir,'checkpoints_vgg')

if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                     mode='training',
                                     batch_size=batch_size,
                                     num_classes=num_classes,
                                     shuffle=True)
    val_data = ImageDataGenerator(val_file,
                                      mode='inference',
                                      batch_size=batch_size,
                                      num_classes=num_classes,
                                      shuffle=False)

    iterator = Iterator.from_structure(tr_data.data.output_types,
                                           tr_data.data.output_shapes)
    next_batch = iterator.get_next()

training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# iterator = tr_data.make_initializable_iterator()
# next_batch = iterator.get_next()


x = tf.placeholder(tf.float32, [batch_size,224,224,3])
y = tf.placeholder(tf.float32,[batch_size, num_classes])
# keep_prob = tf.placeholder(tf.float32)

display_step = 20

model = Vgg16(x,num_classes,weights_path)
score = model.fc8

#list of tf.trainable variables of the layers we want to train
# var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers[i]]

with tf.name_scope('cross_ent'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(score,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# for var in var_list:
#     tf.summary.histogram(var.name, var)
#
# tf.summary.scalar('cross_entropy', loss)

with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# tf.summary.scalar('accuracy', accuracy)

# merged_summary = tf.summary.merge_all()

# writer = tf.summary.FileWriter(filewriter_path)

saver = tf.train.Saver()

train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    # writer.add_graph(sess.graph)

    # model.load_initial_weights(sess)
    # saver.restore(sess, "/path/to/checkpoint/model.ckpt")

    print("{} Start training...".format(datetime.now()))
    # print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      # filewriter_path))
    validation_accuracy = []

    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)

            sess.run(train_op, feed_dict={x: img_batch,
                                          y: label_batch})
                                          # keep_prob: dropout_rate})

            # if step % display_step == 0:
            #     s = sess.run(merged_summary, feed_dict={x: img_batch,
            #                                             y: label_batch})
                                                        # keep_prob: 1.})

                # writer.add_summary(s, epoch*train_batches_per_epoch + step)

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        sess.run(validation_init_op)
        val_acc = 0.
        val_count = 0
        for _ in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)
            acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                y: label_batch})
                                                # keep_prob: 1.})
            val_acc += acc
            val_count += 1
        val_acc /= val_count
        validation_accuracy.append(val_acc)
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       val_acc))
        if val_acc >=  max(validation_accuracy) and val_acc >= 0.95:
#            for filename in os.listdir(checkpoint_path):
#                os.unlink(filename)
            print("{} Saving checkpoint of model...".format(datetime.now()))

            checkpoint_name = os.path.join(checkpoint_path,
                                           'model_epoch'+str(epoch+1)+'.ckpt')
            save_path = saver.save(sess, checkpoint_name)

            print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                           checkpoint_name))
    with open("validation_accuracy_vgg16.txt", "wb") as fp:   #Pickling
        pickle.dump(validation_accuracy, fp)
