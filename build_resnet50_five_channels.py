import tensorflow as tf
from PIL import Image
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

def main():
    global cwd
    cwd='/home/jovyan/mnt/external-images-pvc/ximeng/five_channel_images/'
    Image.open(cwd + 'compound/0.tif.npy')
    #classify_to_subfolder()
    #write_tfrecords()

def classify_to_subfolder():
    df = pd.read_csv("~/scratch-shared/ximeng/dataset_ximeng.csv",';')
    for index, row in df.iterrows():
        if row['type'] == 'compound':
            if os.path.exists(cwd + str(index) + '.tif.npy'):
                shutil.move(cwd + str(index) + '.tif.npy', cwd + 'compound')
        elif row['type'] == 'control':
            if os.path.exists(cwd + str(index) + '.tif.npy'):
                shutil.move(cwd + str(index) + '.tif.npy', cwd + 'control')


def write_tfrecords():
    classes={'control','compound'}
    writer = tf.compat.v1.python_io.TFRecordWriter("/home/jovyan/mnt/external-images-pvc/ximeng/train_five_channels.tfrecords")
    for index,name in enumerate(classes):
        class_path=cwd+name+'/'
        for img_name in os.listdir(class_path): 
            img_path=class_path+img_name 
            img=Image.open(img_path)
            img= img.resize((2160,2160,5))
            print(np.shape(img))
            img_raw=img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            })) 
            writer.write(example.SerializeToString())
    writer.close()




if __name__ == "__main__":
    main()
