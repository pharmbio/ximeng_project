import tensorflow as tf
import tensorflow.keras

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
    biggest_three_families = biggest_three_families(dataframe)
    ##Find the 3 biggest group in the data
    biggest_three_groups = biggest_three_groups(dataframe)
    ##Find the 3 biggest families that do not have overlap between each other in the data
    unique_family_biggest = unique_family_biggest(dataframe)
    ##Find the 3 biggest group that do not have overlap between each other in the data
    unique_group_biggest = unique_group_biggest(dataframe)
    ##get the name list of images
    image_names = image_file_names()
    ##
    for name in image_names:
        sort_images_by_groups(name)
        sort_images_by_families(name)
    
def read_dataframe():
    return pd.read_csv("~/scratch-shared/ximeng/KinaseInhibitorData/dataframe.csv",';')

def biggest_three_families(dataframe):
    return dataframe.family.value_counts().index[0:3]

def biggest_three_groups(dataframe):
    return dataframe.group.value_counts().index[0:3]

def unique_family_biggest(dataframe):
    return dataframe.drop_duplicates('compoundname')['family'].value_counts().index[0:3]

def unique_group_biggest(dataframe):
    return dataframe.drop_duplicates('compoundname')['group'].value_counts().index[0:3]

def image_file_names():
    path_img='/home/jovyan/scratch-shared/ximeng/KinaseInhibitorData/MiSyHo299'
    return os.listdir(path_img)

def sort_images_by_groups(images_names):
    big_group = list(unique_group_biggest)
    if dataframe.type[int(dataframe.index[dataframe.filename==images_names].values)] == 'control':
        shutil.copy(path_img+'/'+ images_names, "/home/jovyan/scratch-shared/ximeng/big_group/control/" + images_names)
    elif dataframe.group[int(dataframe.index[dataframe.filename==images_names].values)] in big_group:
        shutil.copy(path_img+'/'+ images_names, "/home/jovyan/scratch-shared/ximeng/big_group/compound/" + images_names)

def sort_images_by_families(images_names):
    big_family = list(unique_family_biggest)
    if dataframe.type[int(dataframe.index[dataframe.filename==images_names].values)] == 'control':
        shutil.copy(path_img+'/'+ images_names, "/home/jovyan/scratch-shared/ximeng/big_family/control/" + images_names)
    elif dataframe.family[int(dataframe.index[dataframe.filename==images_names].values)] in big_family:
        shutil.copy(path_img+'/'+ images_names, "/home/jovyan/scratch-shared/ximeng/big_family/compound/" + images_names)


if __name__ == "__main__":
    main()



# #screen dataframe
# big_family = list(unique_family_count.index[0:3])
# big_group = list(unique_group_biggest)

# #extract image file names
# path_img='/home/jovyan/scratch-shared/ximeng/KinaseInhibitorData/MiSyHo299'
# img_nms = os.listdir(path_img)

# #sort control and compound 
# #These function didn't work with a error TypeError: only size-1 arrays can be converted to Python scalars

# def sort_control_group(images_names):
#    if dataframe.type[int(dataframe.index[dataframe.filename==images_names].values)] == 'control':
#        shutil.copy(path_img+'/'+ images_names, "/home/jovyan/scratch-shared/ximeng/big_group/control/" + images_names)
#    elif dataframe.group[int(dataframe.index[dataframe.filename==images_names].values)] in big_group:
#        shutil.copy(path_img+'/'+ images_names, "/home/jovyan/scratch-shared/ximeng/big_group/compound/" + images_names)


# def sort_control_family(images_names):
#    if dataframe.type[int(dataframe.index[dataframe.filename==images_names].values)] == 'control':
#        shutil.copy(path_img+'/'+ images_names, "/home/jovyan/scratch-shared/ximeng/big_family/control/" + images_names)
#    elif dataframe.family[int(dataframe.index[dataframe.filename==images_names].values)] in big_family:
#        shutil.copy(path_img+'/'+ images_names, "/home/jovyan/scratch-shared/ximeng/big_family/compound/" + images_names)


# #But if I use only traverse this still got same error but it works,
# for i in img_nms:
#     if dataframe.type[int(dataframe.index[dataframe.filename==i].values)] == 'control':
#         shutil.copy(path_img+'/'+ i, "/home/jovyan/scratch-shared/ximeng/big_group/control/" + i)
#     elif dataframe.group[int(dataframe.index[dataframe.filename==i].values)] in big_group:
#         shutil.copy(path_img+'/'+ i, "/home/jovyan/scratch-shared/ximeng/big_group/compound/" + i)

# for i in img_nms:
#     if dataframe.type[int(dataframe.index[dataframe.filename==i].values)] == 'control':
#         shutil.copy(path_img+'/'+ i, "/home/jovyan/scratch-shared/ximeng/big_family/control/" + i)
#     elif dataframe.family[int(dataframe.index[dataframe.filename==i].values)] in big_family:
#         shutil.copy(path_img+'/'+ i, "/home/jovyan/scratch-shared/ximeng/big_family/compound/" + i)


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

