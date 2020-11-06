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


#tried to be more general but failed
#big_family = unique_family_count.index[0:3].astype(str)
#big_group = unique_group_count.index[0:3].astype(str)

#screen dataframe
family_requried = dataframe['family'].map(lambda x : x in ('EGFR','PIKK','CDK'))
group_requried = dataframe['group'].map(lambda x : x in ('control','TK','Other'))
chosendf = dataframe[family_requried & group_requried]
file_name = str(list(chosendf['filename']))  #maybe array with label here not str?
print(chosendf)


path_img='/home/jovyan/scratch-shared/ximeng/KinaseInhibitorData/MiSyHo299'
ls = os.listdir(path_img)

#can not locate line of chosendf
for i in ls:
    if bool(re.findall("\\b" + i +"\\b", file_name)):
        if (chosendf.loc[chosendf.filename == "\\b" + i +"\\b"],'group') == "control": 
            shutil.copy(path_img+'/'+i , "/home/jovyan/scratch-shared/ximeng/tfpre/control/"+i)
        else:
            shutil.copy(path_img+'/'+i , "/home/jovyan/scratch-shared/ximeng/tfpre/non_control/"+i)



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
