import pandas as pd 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import re

dataframe = pd.read_csv("~/scratch-shared/ximeng/KinaseInhibitorData/dataframe.csv",';')
#1. Write scripts to:
#   a. Find the 3 biggest families in the data
family_count = dataframe.family.value_counts()
print(family_count.head(3))
#   b. Find the 3 biggest group in the data
group_count = dataframe.group.value_counts()
print(group_count.head(3))
#   c. Find the 3 biggest families that do not have overlap between each other in the data
unique_family_count = dataframe.drop_duplicates('compoundname')['family'].value_counts()
print(unique_family_count.head(3))
#   d. Find the 3 biggest group that do not have overlap between each other in the data
unique_group_count = dataframe.drop_duplicates('compoundname')['group'].value_counts()
print(unique_group_count.head(3))

#2. Next I want you to write a CNN that can discern between a control and non-control group using the biggest non-overlapping families/groups you found above
#a. I want you to write this using python, tensorflow2.0
#b. Check in your code regularly
#c. Contact me if you need more help to get going


#screen dataframe
big_family = list(unique_family_count.index[0:3])
big_group = list(unique_group_count.index[0:3])

family_requried = dataframe['family'].map(lambda x : x in big_family)
group_requried = dataframe['group'].map(lambda x : x in big_group)
#subset dataframe of biggest families/groups
chosendf = dataframe[family_requried & group_requried]
#extract only needed filename and group, can be deleted later when rest works
file_name = chosendf[['filename','group']]
#extract image file names
path_img='/home/jovyan/scratch-shared/ximeng/KinaseInhibitorData/MiSyHo299'
img_nms = os.listdir(path_img)
img_nms = [i.replace('.png','') for i in img_nms]

#sort images into different classes
for index,row in file_name.iterrows():
    if row[0] in img_nms:
        if row[1] == "control": 
            shutil.copy(path_img+'/'+ row[0] + '.png' , "/home/jovyan/scratch-shared/ximeng/tfpre/control/"+row[0] + '.png')
        else:
            shutil.copy(path_img+'/'+ row[0] + '.png' , "/home/jovyan/scratch-shared/ximeng/tfpre/non_control/"+row[0] + '.png')

#Then every thing goes into non_control group(this one has a weird error now, but similar partin prac.py would work), this made me frustrate.
#BUT, I just took a detailed look at the extracted datarame(file_name), it looks like all chosen data are TK group??
#Now I feel I WASTED WHOLE week...

#Or have I made some mistake?



#the rest part can be ignore so far
cwd='/home/jovyan/scratch-shared/ximeng/tfpre/'
classes={'control','non_control'}
writer= tf.python_io.TFRecordWriter("msh_test.tfrecords")
for index,name in enumerate(classes):
    class_path=cwd+name+'/'
    for img_name in os.listdir(class_path): 
        img_path=class_path+img_name 
        img=Image.open(img_path)
        img= img.resize((299,299))
        print(np.shape(img))
        img_raw=img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        })) 
        writer.write(example.SerializeToString())
 
writer.close()
