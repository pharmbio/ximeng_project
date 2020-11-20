#import tensorflow as tf

from PIL import Image
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

##This is a practice
def main():
    ##read dataframe file
    dataframe = read_dataframe()
    ##Find the 3 biggest families in the data
    biggest_three_family = biggest_three_families(dataframe)
    ##Find the 3 biggest group in the data
    biggest_three_group = biggest_three_groups(dataframe)
    ##Find the 3 biggest families that do not have overlap between each other in the data
    unique_families_biggest = unique_family_biggest(dataframe)
    ##Find the 3 biggest group that do not have overlap between each other in the data
    unique_groups_biggest = unique_group_biggest(dataframe)
    ##get the name list of images
    image_names = image_file_names()
    ##
    for name in image_names:
         sort_images_by_groups(name,dataframe)
         sort_images_by_families(name,dataframe)
    
    print('Good Bye')


def read_dataframe():
    return pd.read_csv("~/scratch-shared/ximeng/KinaseInhibitorData/dataframe.csv",';')

def biggest_three_families(df):
    return df.family.value_counts().index[0:3]

def biggest_three_groups(df):
    return df.group.value_counts().index[0:3]

def unique_family_biggest(df):
    uni_family = df.drop_duplicates('compoundname')['family'].value_counts().index[0:3]
    global big_family
    big_family = list(uni_family)
    return uni_family
def unique_group_biggest(df):
    uni_group = df.drop_duplicates('compoundname')['group'].value_counts().index[0:3]
    global big_group
    big_group = list(uni_group)
    return uni_group
def image_file_names():
    global image_path
    image_path='/home/jovyan/scratch-shared/ximeng/KinaseInhibitorData/MiSyHo299'
    return os.listdir(image_path)

def sort_images_by_groups(images_names,df):
    if df.type[(df.index[df.filename == images_names].values).astype(int)] is 'control':
        shutil.copy(image_path+'/'+ images_names, "/home/jovyan/scratch-shared/ximeng/big_group/control/" + images_names)
    elif df.group[(df.index[df.filename == images_names].values).astype(int)].any in big_group:
        shutil.copy(image_path+'/'+ images_names, "/home/jovyan/scratch-shared/ximeng/big_group/compound/" + images_names)

def sort_images_by_families(images_names,df):
    if df.type[(df.index[df.filename == images_names].values).astype(int)] is 'control':
        shutil.copy(image_path+'/'+ images_names, "/home/jovyan/scratch-shared/ximeng/big_family/control/" + images_names)
    elif df.family[(df.index[df.filename == images_names].values).astype(int)].any in big_family:
        shutil.copy(image_path+'/'+ images_names, "/home/jovyan/scratch-shared/ximeng/big_family/compound/" + images_names)


if __name__ == "__main__":
    main()



# #building models based on biggest groups
# cwd='/home/jovyan/scratch-shared/ximeng/big_group/'
# classes={'control','compound'}
# writer = tf.compat.v1.python_io.TFRecordWriter("/home/jovyan/scratch-shared/ximeng/train_big_group.tfrecords")
# for index,name in enumerate(classes):
#     class_path=cwd+name+'/'
#     for img_name in os.listdir(class_path): 
#         img_path=class_path+img_name 
#         img=Image.open(img_path)
#         img= img.resize((299,299,3))
#         print(np.shape(img))
#         img_raw=img.tobytes()
#         example = tf.train.Example(features=tf.train.Features(feature={
#             "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
#             'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
#         })) 
#         writer.write(example.SerializeToString())
# writer.close()

