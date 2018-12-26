"""Model module

"""
###
from Config import NODES_FC1,NODES_FC2,OUTPUT_CLASSES
from Utils import start_log
import tensorflow as tf
###


#ALEX_NET227
def Model_ALEXNET(x,y_true,hold_prob1,hold_prob2,logger):
    logger.info("Model Structure: ")
    ###CONV_LAYER-1
    #Weights-Layer-1
    w_1 = tf.Variable(tf.truncated_normal([11,11,3,96], stddev=0.01))
    #Bias-Layer-1
    b_1 = tf.Variable(tf.constant(0.0, shape=[[11,11,3,96][3]]))
    #Convolution-Layer-1
    c_1 = tf.nn.conv2d(x, w_1,strides=[1, 4, 4, 1], padding='VALID')
    #Adding bias
    c_1 = c_1 + b_1
    #Applying RELU
    c_1 = tf.nn.relu(c_1)
                                    
    logger.info(c_1)
    ##Max-Pool-Layer-1
    p_1 = tf.nn.max_pool(c_1, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='VALID')
    logger.info(p_1)

    ###CONV_LAYER 2
    w_2 = tf.Variable(tf.truncated_normal([5,5,96,256], stddev=0.01))
    b_2 = tf.Variable(tf.constant(1.0, shape=[[5,5,96,256][3]]))
    c_2 = tf.nn.conv2d(p_1, w_2,strides=[1, 1, 1, 1], padding='SAME')
    c_2 = c_2 + b_2
    c_2 = tf.nn.relu(c_2)

    logger.info(c_2)
    ##Max-Pool-Layer-2
    p_2 = tf.nn.max_pool(c_2, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='VALID')
    logger.info(p_2)

    ###CONV_LAYER 3
    w_3 = tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.01))
    b_3 = tf.Variable(tf.constant(0.0, shape=[[3, 3, 256, 384][3]]))
    c_3 = tf.nn.conv2d(p_2, w_3,strides=[1, 1, 1, 1], padding='SAME')
    c_3 = c_3 + b_3
    c_3 = tf.nn.relu(c_3)

    logger.info(c_3)

    ###CONV_LAYER 4
    w_4 = tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.01))
    b_4 = tf.Variable(tf.constant(0.0, shape=[[3, 3, 384, 384][3]]))
    c_4 = tf.nn.conv2d(c_3, w_4,strides=[1, 1, 1, 1], padding='SAME')
    c_4 = c_4 + b_4
    c_4 = tf.nn.relu(c_4)

    logger.info(c_4)

    ###CONV_LAYER 5
    w_5 = tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.01))
    b_5 = tf.Variable(tf.constant(0.0, shape=[[3, 3, 384, 256][3]]))
    c_5 = tf.nn.conv2d(c_4, w_5,strides=[1, 1, 1, 1], padding='SAME')
    c_5 = c_5 + b_5
    c_5 = tf.nn.relu(c_5)

    logger.info(c_5)

    ##Max-Pool-Layer-3
    p_3 = tf.nn.max_pool(c_5, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='VALID')
    logger.info(p_3)

    #Flattening
    flattened = tf.reshape(p_3,[-1,6*6*256])
    logger.info(flattened)

    ###FC_Layer-1
    #Getting input nodes in FC layer 1
    input_size = int( flattened.get_shape()[1] )
    #Weights for FC Layer 1
    w1_fc = tf.Variable(tf.truncated_normal([input_size, NODES_FC1], stddev=0.01))
    #Bias for FC Layer 1
    b1_fc = tf.Variable( tf.constant(1.0, shape=[NODES_FC1] ) )
    #Summing Matrix calculations and bias
    s_fc1 = tf.matmul(flattened, w1_fc) + b1_fc
    #Applying RELU
    s_fc1 = tf.nn.relu(s_fc1)

    #Dropout Layer 1
    s_fc1 = tf.nn.dropout(s_fc1,keep_prob=hold_prob1)

    logger.info(s_fc1)

    ###FC_Layer-2
    w2_fc = tf.Variable(tf.truncated_normal([NODES_FC1, NODES_FC2], stddev=0.01))
    b2_fc = tf.Variable( tf.constant(1.0, shape=[NODES_FC2] ) )
    s_fc2 = tf.matmul(s_fc1, w2_fc) + b2_fc
    s_fc2 = tf.nn.relu(s_fc2)
    logger.info(s_fc2)

    #Dropout Layer 2
    s_fc2 = tf.nn.dropout(s_fc2,keep_prob=hold_prob2)

    ###FC_Layer-3
    w3_fc = tf.Variable(tf.truncated_normal([NODES_FC2,OUTPUT_CLASSES], stddev=0.01))
    b3_fc = tf.Variable( tf.constant(1.0, shape=[OUTPUT_CLASSES] ) )
    y_pred = tf.matmul(s_fc2, w3_fc) + b3_fc
    logger.info(y_pred)


    return y_pred