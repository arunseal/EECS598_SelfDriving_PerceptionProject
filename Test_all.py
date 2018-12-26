from Utils import get_data_test,write_csv,start_log
from Config import IMG_SIZE_ALEXNET,TEST_FILE_PATH,MODEL_NAME,BATCH_SIZE,MODEL_SAVE_PATH,CSV_SAVE_PATH,DEBUG
###Start-logging
logger=start_log(file_path='./Logs',log_name="Test")

import os  
import time
import sys
import numpy as np              
import tensorflow as tf
###

def Test():
    """Test function
    """
    logger.info("Getting data")
    ##Get test Data
    test_x,data_id=get_data_test(TEST_FILE_PATH,logger,debug=DEBUG)
    total=len(test_x)
    logger.info("Preprocessing Data: ")
    ##verify data shape
    try:
        # assert(test_x.shape==(test_x.shape[0],IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET,3))
        assert(len(test_x[0])==IMG_SIZE_ALEXNET)
    except AssertionError:
        logger.error("Data shape does not meet requirement")
    logger.info("Preprocessed Data")
    
    #Check flag status
    # if flag is not "Fail":
    # try:#Starting Tensorflow session
    logger.info("Starting tensorflow session")
    with tf.Session() as sess:
        batch_size=BATCH_SIZE
        logger.info("Importing saved model")
        ##Importing Saved Model
        # try:
        # print(MODEL_SAVE_PATH+'/'MODEL_NAME)
        saver = tf.train.import_meta_graph(MODEL_SAVE_PATH+'/'+MODEL_NAME)
        # saver=tf.train.import_meta_graph('./checkpoints/CNN_BI.ckptmodel.meta')
        saver.restore(sess,tf.train.latest_checkpoint(MODEL_SAVE_PATH))
        graph=tf.get_default_graph()
        # except:
        # logger.error("Error retrieving saved model")
            ##Importing tensors
        # try:
        x_input=graph.get_tensor_by_name("x_input:0")
        is_training=graph.get_tensor_by_name("is_training:0")
        # y_true=graph.get_tensor_by_name("y_true:0")
        # hold_prob1 = graph.get_tensor_by_name("hold_prob1:0")
        # hold_prob2 = graph.get_tensor_by_name("hold_prob2:0")
        # acc=graph.get_tensor_by_name("acc:0")
        # cross_entropy=graph.get_tensor_by_name("cross_entropy:0")
        pred_classes=graph.get_tensor_by_name("pred_classes:0")
        # pred_probs=graph.get_tensor_by_name("pred_probs:0")
        # except:
            # logger.error("Error getting tensors")

        pred_labels = []
        start=0
        logger.info("Starting testing")
        time_start_test=time.time()
        ##TESTING STARTS
        while(start<total):
            if (start+batch_size)<=total:
                end=start+batch_size
            else:
                end=total
            test_batch_x=test_x[start:end]
            test_pred_class = sess.run(pred_classes,
            feed_dict ={x_input: test_batch_x,is_training:False})
            # print(test_pred_class)
            ##Count defects, not_defects, FP and FN
            test_pred_class=np.asarray(test_pred_class)
            ###Write out probabilities and predicted classes to output csv file
            write_csv(CSV_SAVE_PATH,test_pred_class,data_id,count=start,logger=logger)
            # print(test_pred_class.shape[1])
            start=start+batch_size
        logger.info("Finished Testing")


                ###END TESTING
    # except Exception as error:
    #     logger.error("Process Failed. See logs for errors %s", error.__class__.__name__)

if __name__=='__main__':
    Test()





    
