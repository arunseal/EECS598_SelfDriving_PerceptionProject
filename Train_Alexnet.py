"""Train module

"""
###
from Utils import get_data_train_val,get_data_test,start_log
from Config import TRAIN_FILE_PATH,TEST_FILE_PATH,IMG_SIZE_ALEXNET,BATCH_SIZE,EPOCHS,LEARNING_RATE,OUTPUT_CLASSES,MODEL_SAVE_PATH,DEBUG
from Model_ALEXNET import Model_ALEXNET
from sklearn.model_selection import train_test_split

###logging
logger=start_log(file_path='./Logs',log_name="Train")
###

import os
import time
import sys

import numpy as np
import tensorflow as tf
###

###turns off tensorflow logging on the console
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ["CUDA_VISIBLE_DEVICES"]= '-1'

def Train():
    """Training function for classifier

    """
    # try:
    X_full,y_full,data_id_train= get_data_train_val(file_path=TRAIN_FILE_PATH,logger=logger,image_size=IMG_SIZE_ALEXNET,debug=DEBUG)
    # steps = X.shape[0]   
    # steps=len(X_full)
    # remaining = steps % BATCH_SIZE
    # except:
    #     logger.error("Could not get train data")

    X, val_x, y, val_y = train_test_split(X_full, y_full, test_size=0.2, random_state=666)
    total=len(val_x)
    steps=len(X)
    remaining = steps % BATCH_SIZE
    # try:
    #     test_x,test_y,data_id_test = get_data_train_test(file_path=TEST_FILE_PATH,logger=logger)
    #     total=len(test_x)
    # except:
    #     logger.error("Could not get test data")

    logger.info("Preprocessing Data: ")
    ##verify data shapes
    try:
        assert len(X[0])==IMG_SIZE_ALEXNET
        assert len(val_x[0])==IMG_SIZE_ALEXNET
    except AssertionError:
        logger.error("Data shape does not meet requirement")
    logger.info("Preprocessed Data")

    #Resetting graph
    logger.info("Reset Graph")
    tf.reset_default_graph()

    logger.info("Setting Placeholders")
    #Defining Placeholders
    x = tf.placeholder(tf.float32,shape=[None,IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET,3],name='x')
    y_true = tf.placeholder(tf.float32,shape=[None,OUTPUT_CLASSES],name='y_true')
    hold_prob1 = tf.placeholder(tf.float32,name='hold_prob1')
    hold_prob2 = tf.placeholder(tf.float32,name='hold_prob2')

    logger.info("Getting Predicted Probs")
    #Predicted Probs
    y_pred=Model_ALEXNET(x,y_true,hold_prob1,hold_prob2,logger)

    logger.info("Finding Loss")
    #Defining loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y_pred),name='cross_entropy')

    logger.info("Running Optimizer")
    #Defining objective
    train_opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)

    logger.info("Getting accuracy metrics")
    #Defining probabilities and accuracy
    pred_probs=tf.nn.softmax(y_pred,name='pred_probs')
    pred_classes=tf.argmax(pred_probs,1,name='pred_classes')
    matches = tf.equal(pred_classes,tf.argmax(y_true,1))
    acc = tf.reduce_mean(tf.cast(matches,tf.float32),name='acc')
    
    init = tf.global_variables_initializer()
    try:
        saver = tf.train.Saver(max_to_keep=1)
    except:
        logger.warning("Error creating training saved model")
    test_acc_list_total=[]
    test_loss_list_total=[]

    logger.info("Starting Tensorflow session")
    # try:#Starting TF session
    with tf.Session() as sess:
        logger.info("Initializing global variables")
        sess.run(init)
        logger.info("Started Training")
        for i in range(EPOCHS):
            ##TRAINING STARTS
            time_start=time.time()
            for j in range(0,steps-remaining,BATCH_SIZE):
                sess.run([train_opt,cross_entropy],feed_dict={x:X[j:j+BATCH_SIZE] , y_true:y[j:j+BATCH_SIZE],hold_prob1:0.5,hold_prob2:0.5})
            logger.info("Epoch: %s, Trained in time: %s",i,time.time()-time_start)
            ##TRAINING ENDS

            time_start_test=time.time()
            test_def_list,test_not_def_list,test_fp_list,test_fn_list=0,0,0,0
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
                test_def_true,test_not_def_true,test_fp,test_fn=0,0,0,0
                test_batch_x=val_x[start:end]
                test_batch_y=val_y[start:end] 
                acc_on_test,loss_on_test,preds_test,test_pred_class = sess.run([acc,cross_entropy,pred_probs,pred_classes],
                feed_dict ={x: test_batch_x,y_true:test_batch_y,hold_prob1:1.0,hold_prob2:1.0})
                #Counting defects,non-defects,fp and fn
                test_acc_list.append(acc_on_test)
                test_loss_list.append(loss_on_test)

                start+=BATCH_SIZE

            test_acc_ = round(np.mean(test_acc_list),5)
            test_loss_ = round(np.mean(test_loss_list),5)
            test_acc_list_total.append(test_acc_)
            test_loss_list_total.append(test_loss_)
            logger.info("Test Results: ")
            logger.info("Epoch: %i, Accuracy: %f, Loss: %f, Time: %f",i,test_acc_,test_loss_,time.time()-time_start_test)



            #Saving Model
            if test_acc_>=max(test_acc_list_total) and test_acc_>=0.935:
                try:
                    if not os.path.exists(MODEL_SAVE_PATH):
                        os.makedirs(MODEL_SAVE_PATH)
                    # save_file_name=MODEL_SAVE_PATH +'/'+str(round(test_acc_,4))
                    save_file_name=MODEL_SAVE_PATH +'/'+'model'
                    saver.save(sess,save_file_name)
                    logger.info('Saved model: %s',save_file_name)
                except:
                    logger.warning("Error saving model")

            del test_acc_,test_loss_
                ###TESTING ENDS

    # except Exception as error:
        # logger.error("Process Failed. See logs for errors %s", error.__class__.__name__)

if __name__=='__main__':
    Train()