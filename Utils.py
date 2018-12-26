"""Utils Module

"""
###
from Config import IMG_SIZE_ALEXNET,IMG_SIZE_RESNET,IMG_SIZE_VGG
import os
import csv
import time
import sys
import logging
import datetime
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
import cv2
###
from Config import TRAIN_LABELS_PATH

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def get_data_train_val(file_path,logger,image_size,debug=False):
    
    start=time.time()
    logger.info("Started retreiving Data")
    logger.info("File path: %s",file_path)
    if file_path is None:
        logger.error("Training File path does not exist") #Check Params
    labels=[]
    data=[]
    data_id=[]
    folders=[]
    imgs=[]
    # try:
    csv_file=csv.reader(open(TRAIN_LABELS_PATH,'r'))
    # except FileNotFoundError:
        # logger.error("Csv labels File not found")

    next(csv_file)
    count_labels=0
    for row in csv_file:
        # next(row)
        try:
            if row[1]=='0':
                labels.append(0)
                count_labels+=1
            elif row[1]=='1':
                labels.append(1)
                count_labels+=1
            elif row[1]=='2':
                labels.append(2)
                count_labels+=1
            data_id.append(row[0])
            folder,img=row[0].split('/')
            folders.append(folder)
            imgs.append(img)
            if count_labels==100 and debug:
                break
            # if count_labels==500:
            #     break
        except:
            logger.error("Csv file format is not supported")
        print("read label: ",end='\r')#Delete Later


    labels=np.asarray(to_categorical(labels),dtype=np.int32)

    # folder,img=str(data_id).split('/')
    data_id=np.asarray(data_id)


    count_data=0
    for each in range(len(data_id)):
        path=(str(folders[each])+"/"+str(imgs[each]+"_image.jpg"))#May need revision depending on format
        image=cv2.imread((file_path+'/'+path))

        if image is None:
            logger.warning("Could not read image: %s",path)#Check path
        if image  is not None:
            image=cv2.resize(image,(image_size,image_size))
            # image=adjust_gamma(image,gamma=1.5)
            data.append(image)
            del image
            count_data+=1
            if count_data==100 and debug:
                break
            print("read image: ",count_data,end='\r')#Delete Later


    try:
        logger.info("Data shape: %s",len(data))
        logger.info("Labels shape: %s", labels.shape)
        logger.info("Loaded training data in: %f", time.time()-start)
        assert count_data==count_labels
    except AssertionError:
        logger.error("Data and label length does not match")

    return data,labels,data_id


def get_data_test(file_path,logger,image_size,debug=False):

    start=time.time()
    logger.info("Started retreiving Testing Data")
    logger.info("Images path: %s",file_path)
    data=[]
    data_id=[]
    if file_path is None:
        logger.error("Testing Images path does not exist")#Check Params
    folder_path=os.listdir(file_path)
    count_data=0

    for folder in folder_path:
        path=file_path+folder
        img_path=os.listdir(path)
        for img in img_path:
            if img.endswith('.jpg'):
                image=cv2.imread(path+'/'+img)
                data_id.append(str(folder)+'/'+str(img))
                
                if image is None:
                    logger.warning("Could not read image: %s",img)#Check Path
                if image  is not None:
                    image=cv2.resize(image,(image_size,image_size))
                    data.append(image)
                    # del image
                    count_data+=1
            if count_data==100 and debug:
                break
            print("read image: ",count_data,end='\r')#Delete Later

    # data = np.asarray(data,dtype=np.uint8)
    data_id=np.asarray(data_id)

    try:
        logger.info("Total count: %d",count_data)
        logger.info("Data shape: %s",len(data))
        logger.info("Loaded Data in: %f", time.time()-start)
        assert count_data==len(data_id)
    except AssertionError:
        logger.error("Data and file_name length does not match")
    return data,data_id

#Writes output prediction file
# def write_csv(save_path,test_pred_class,data_id,logger):

#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     file_save_path=save_path
#     # try:
#         # if count==0:
#     file_save = open(file_save_path,'w')
#     header="guid/image,label"
#     file_save.write(header)
#     file_save.write("\n")
#         # else:
#         #     file_save= open(file_save_path,'a')
#     # except Exception as error:
#     #     logger.error("Error opening _results.csv, error %s",error.__class__.__name__)
    
#     # try:
#     size=len(test_pred_class)
#     for row in range(size):
#         row_write=str(data_id[row])+","+str(test_pred_class[row])
#         file_save.write(row_write)
#         file_save.write("\n")
#     # except Exception as error:
#     #     logger.error("Error writing out data, error: %s",error.__class__.__name__)

def write_csv(file_path,test_pred_class,data_id,count,logger):
    """Writes output csv File

    """
    # file_path=PREDS_SAVE_PATH
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_save_path=file_path+'/'+'out_labels.csv'
    # file_save_path=file_path
    try:
        if count==0:
            file_save = open(file_save_path,'w')
            header="guid/image,label"
            file_save.write(header)
            file_save.write("\n")
        else:
            file_save= open(file_save_path,'a')
    except Exception as error:
        logger.error("Error opening _results.csv, error %s",error.__class__.__name__)
    
    # try:
    # size=test_pred_class.shape[1]
    size=len(test_pred_class)
    for row in range(size):
        row_write=str(data_id[count+row]).replace("_image.jpg","")+","+str(test_pred_class[row])
        file_save.write(row_write)
        file_save.write("\n")
    # except Exception as error:
    #     logger.error("Error writing out data, error: %s",error.__class__.__name__)
    


#Logging Function
def start_log(file_path,log_name=__name__):
    """Logging Function
    """
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_name=file_path+'/'+datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.txt'
    file_handler = logging.FileHandler(filename=file_name)
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]

    logging.basicConfig(
        level=logging.INFO, 
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )
    stdout_handler.level=logging.INFO

    logger = logging.getLogger(log_name)
    return logger
    




        
    
