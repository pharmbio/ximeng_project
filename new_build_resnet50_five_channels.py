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
    train_dataset = write_datasets()
    train_dataset = train_dataset.map(load_image_label)
    final_model = build_model(train_dataset)


def classify_to_subfolder():
    df = pd.read_csv("~/scratch-shared/ximeng/dataset_ximeng.csv",';')
    for index, row in df.iterrows():
        if row['type'] == 'compound':
            if os.path.exists(cwd + str(index) + '.tif'):
                shutil.move(cwd + str(index) + '.tif', cwd + 'compound')
        elif row['type'] == 'control':
            if os.path.exists(cwd + str(index) + '.tif'):
                shutil.move(cwd + str(index) + '.tif', cwd + 'control')


def write_datasets():
    from glob import glob
    data_directory = glob('/home/jovyan/mnt/external-images-pvc/ximeng/five_channel_images/*/*.npy')
    train_dataset = tf.data.Dataset.list_files('/home/jovyan/mnt/external-images-pvc/ximeng/five_channel_images/*/*.npy')
    
    # train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    # data_directory,
    # labels='inferred',
    # label_mode = 'categorical',
    # shuffle=True,
    # seed=123,
    # image_size=(2160, 2160),)
    return train_dataset

def load_image_label(path):
    
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image)
    label = tf.strings.split(path, os.path.sep)[-2]
    return image, label

def build_model(train_dataset):
    input_layer = tf.keras.layers.Input(shape=(2160,2160,5))
    second_layer = layers.Conv2D(3, 1937, 1, activation='relu')(input_layer)
    resnet50_layers = tf.keras.applications.ResNet50(include_top=True, weights='imagenet')(second_layer)
    outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(resnet50_layers)
    model = tf.keras.Model(inputs=input_layer, outputs=outputs)
    model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.summary()

    model.fit(train_dataset, epochs=10)
    callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]
    return model


if __name__ == "__main__":
    main()
