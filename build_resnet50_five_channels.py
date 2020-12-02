import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from PIL import Image
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

def main():
    global cwd
    cwd='/home/jovyan/mnt/external-images-pvc/ximeng/five_channel_images/'
    #classify_to_subfolder()
    #write_tfrecords()
    build_model("/home/jovyan/mnt/external-images-pvc/ximeng/train_five_channels.tfrecords")

def classify_to_subfolder():
    df = pd.read_csv("~/scratch-shared/ximeng/dataset_ximeng.csv",';')
    for index, row in df.iterrows():
        if row['type'] == 'compound':
            if os.path.exists(cwd + str(index) + '.tif'):
                shutil.move(cwd + str(index) + '.tif', cwd + 'compound')
        elif row['type'] == 'control':
            if os.path.exists(cwd + str(index) + '.tif'):
                shutil.move(cwd + str(index) + '.tif', cwd + 'control')


def write_tfrecords():
    classes={'control','compound'}
    writer = tf.compat.v1.python_io.TFRecordWriter("/home/jovyan/mnt/external-images-pvc/ximeng/train_five_channels.tfrecords")
    for index,name in enumerate(classes):
        class_path=cwd+name+'/'
        for image_name in os.listdir(class_path): 
            if image_name.endswith('.npy'):
                image_path=class_path+image_name 
                image=np.load(image_path)
                #image= image.resize((5,2160,2160))
                print(np.shape(image))
                image_raw=image.tobytes()
                working_image = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
                }))
                writer.write(working_image.SerializeToString())
    writer.close()

def build_model(tfrecord_file_path):
    input_dataset = tf.data.TFRecordDataset(tfrecord_file_path)
    model = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet', input_tensor=None,
    pooling=max)
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.fit(input_dataset, epochs=10)
    callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]

if __name__ == "__main__":
    main()
