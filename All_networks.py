import tensorflow as tf
import tensornets as nets
from Utils import get_data_train_val,get_data_test,start_log
from Config import TRAIN_FILE_PATH,TEST_FILE_PATH,IMG_SIZE_VGG,BATCH_SIZE,EPOCHS,LEARNING_RATE,OUTPUT_CLASSES,MODEL_SAVE_PATH,DEBUG
from sklearn.model_selection import train_test_split

###logging
logger=start_log(file_path='./Logs',log_name="Train")
###

import os
import time
import sys

import numpy as np


def Train():
    """Training function for classifier

    """
    X_full,y_full,data_id_train= get_data_train_val(file_path=TRAIN_FILE_PATH,logger=logger,debug=DEBUG)
    X_full -= np.mean(X_full, axis = 0)
    X_full /= np.std(X_full, axis = 0)
    X, val_x, y, val_y = train_test_split(X_full, y_full, test_size=0.2, random_state=666)
    total=len(val_x)
    steps=len(X)
    remaining = steps % BATCH_SIZE

    logger.info("Preprocessing Data: ")
    ##verify data shapes
    try:
        assert len(X[0])==IMG_SIZE_VGG
        assert len(val_x[0])==IMG_SIZE_VGG
    except AssertionError:
        logger.error("Data shape does not meet requirement")
    logger.info("Preprocessed Data")

    #Resetting graph
    logger.info("Reset Graph")
    tf.reset_default_graph()

    logger.info("Setting Placeholders")
    #Defining Placeholders
    # x = tf.placeholder(tf.float32,shape=[None,IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET,3],name='x')
    # y_true = tf.placeholder(tf.float32,shape=[None,OUTPUT_CLASSES],name='y_true')
    # hold_prob1 = tf.placeholder(tf.float32,name='hold_prob1')
    # hold_prob2 = tf.placeholder(tf.float32,name='hold_prob2')

    x_input = tf.placeholder(tf.float32, [None, IMG_SIZE_VGG, IMG_SIZE_VGG, 3],name='x_input')
    y_true = tf.placeholder(tf.float32, [None, OUTPUT_CLASSES],name='y_true')
    is_training=tf.placeholder(tf.bool,name='is_training')

    # VGG19 returns the last layer (softmax)
    # model to give the name
    # logits = nets.VGG19(inputs, is_training=is_training, classes=OUTPUT_CLASSES)
    # model = tf.identity(logits, name='logits')

    model=nets.VGG19(x_input,is_training=is_training,classes=OUTPUT_CLASSES)
    logits=tf.identity(model,name='logits')

    # loss function applied to the last layer
    # train on the loss (Adam Optimizer is used)
    cross_entropy = tf.losses.softmax_cross_entropy(y_true, logits)
    train_opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)

    # for measuring accuracy after forward passing
    pred_classes=tf.argmax(logits,1,name='pred_classes')
    correct_pred = tf.equal(pred_classes, tf.argmax(y_true, 1),name='correct_pred')
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    
    init = tf.global_variables_initializer()
    try:
        saver = tf.train.Saver(max_to_keep=1)
    except:
        logger.warning("Error creating training saved model")
    test_acc_list_total=[]
    test_loss_list_total=[]

    model_saved=False
    logger.info("Starting Tensorflow session")
    # try:#Starting TF session
    with tf.Session() as sess:
        logger.info("Initializing global variables")
        sess.run(init)
        logger.info("Started Training")
        # Loading the parameters
        # nets.pretrained(model)
        sess.run(model.pretrained())
        for i in range(EPOCHS):
            ##TRAINING STARTS
            time_start=time.time()
            for j in range(0,steps-remaining,BATCH_SIZE):
                sess.run(train_opt,feed_dict={x_input:X[j:j+BATCH_SIZE] , y_true:y[j:j+BATCH_SIZE],is_training:True})
            logger.info("Epoch: %s, Trained in time: %s",i,time.time()-time_start)
            ##TRAINING ENDS

            time_start_test=time.time()
            test_acc_list = []
            test_loss_list = []
            pred_labels = []
            start=0

            ##TESTING STARTS
            while(start<total):
                if (start+BATCH_SIZE)<=total:
                    end=start+BATCH_SIZE
                else:
                    end=total
                test_batch_x=val_x[start:end]
                test_batch_y=val_y[start:end] 
                acc_on_test= sess.run(acc,feed_dict ={x_input: test_batch_x,y_true:test_batch_y,is_training:False})
                #Counting defects,non-defects,fp and fn
                test_acc_list.append(acc_on_test)
                # test_loss_list.append(loss_on_test)

                start+=BATCH_SIZE

            test_acc_ = round(np.mean(test_acc_list),5)
            # test_loss_ = round(np.mean(test_loss_list),5)
            test_acc_list_total.append(test_acc_)
            # test_loss_list_total.append(test_loss_)
            logger.info("Test Results: ")
            # logger.info("Epoch: %i, Accuracy: %f, Loss: %f, Time: %f",i,test_acc_,test_loss_,time.time()-time_start_test)
            logger.info("Epoch: %i, Accuracy: %f, Time: %f",i,test_acc_,time.time()-time_start_test)



            #Saving Model
            if test_acc_>=max(test_acc_list_total) and test_acc_>=0.90:
                try:
                    if not os.path.exists(MODEL_SAVE_PATH):
                        os.makedirs(MODEL_SAVE_PATH)
                    # save_file_name=MODEL_SAVE_PATH +'/'+str(round(test_acc_,4))
                    save_file_name=MODEL_SAVE_PATH +'/'+'model'
                    saver.save(sess,save_file_name)
                    logger.info('Saved model: %s',save_file_name)
                    model_saved=True
                except:
                    logger.warning("Error saving model")

            if i==EPOCHS-1 and model_saved==False:
                try:
                    if not os.path.exists(MODEL_SAVE_PATH):
                        os.makedirs(MODEL_SAVE_PATH)
                    # save_file_name=MODEL_SAVE_PATH +'/'+str(round(test_acc_,4))
                    save_file_name=MODEL_SAVE_PATH +'/'+'model'
                    saver.save(sess,save_file_name)
                    logger.info('Saved model: %s',save_file_name)
                    model_saved=True
                except:
                    logger.warning("Error saving model")

            # del test_acc_,test_loss_
            del test_acc_
                ###TESTING ENDS

    # except Exception as error:
        # logger.error("Process Failed. See logs for errors %s", error.__class__.__name__)

if __name__=='__main__':
    Train()