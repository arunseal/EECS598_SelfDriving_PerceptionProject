import pandas as pd
import os
from pathlib import Path
import numpy as np

def append_path(x,data_directory):
    dir_image = x.split('/')
    path = os.path.join(data_directory,dir_image[0],dir_image[1]+'_image.jpg')
    # return Path(path)
    # return path.encode('unicode_escape').decode()
    return path

def read_classes(x,data_directory):
    dir_bbox = x.split('/')
    path = os.path.join(data_directory,dir_bbox[0],dir_bbox[1]+'_bbox.bin')
    try:
        bbox = np.fromfile(path, dtype=np.float32)
        class_id = int(bbox[-2])
    except FileNotFoundError:
        class_id = int(0)
    # bbox = bbox.reshape([-1, 11])
    return class_id


df = pd.read_csv('labels.csv')
current_dir = os.getcwd()
data_dir = os.path.join(current_dir,'deploy','trainval')

df['class_id'] = df['guid/image'].apply(lambda x: read_classes(x,data_dir))
df['guid/image'] = df['guid/image'].apply(lambda x: append_path(x,data_dir))


train_val_split = 0.8
train_df = df.sample(frac=train_val_split,random_state=10)
val_df= df.drop(train_df.index)

train_df.to_csv('train.csv',index=False)
val_df.to_csv('validation.csv',index=False)

df1 = pd.read_csv('train.csv')
my_list = []
for i in df1.index:
    path = df1.iloc[i]['guid/image']

    if not os.path.exists(path):
        # print(path)
        # print(Path(path))
        my_list.append(i)
df_1 = df1.drop(my_list)
print('TOTAL TRAIN IMAGES: ', len(df1.index))
print('TRAINABLE IMAGES IN TRAIN SET: ', len(df1.index)-len(my_list))
df_1.to_csv('train_v2.csv',index=False)

df2 = pd.read_csv('validation.csv')
my_list = []
for i in df2.index:
    path = df2.iloc[i]['guid/image']

    if not os.path.exists(path):
        # print(path)
        # print(Path(path))
        my_list.append(i)
df_2 = df2.drop(my_list)
print('TOTAL VALIDATION IMAGES: ', len(df2.index))
print('USABLE IMAGES IN VALIDATION SET: ', len(df2.index)-len(my_list))
df_2.to_csv('validation_v2.csv',index=False)
