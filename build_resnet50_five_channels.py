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
    parsed_image_dataset = read_tfrecord_file()
    final_model = build_model(parsed_image_dataset)
    model_probability(final_model,parsed_image_dataset)


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
                image_raw=np.load(image_path).astype(int)
                image_raw=image_raw.tobytes()
                working_image = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
                }))
                writer.write(working_image.SerializeToString())
    writer.close()

def _parse_function(example_proto):
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),}
    return tf.io.parse_single_example(example_proto, image_feature_description)

def read_tfrecord_file():
    raw_dataset = tf.keras.preprocessing.image_dataset_from_directory('/home/jovyan/mnt/external-images-pvc/ximeng/five_channel_images/', labels='inferred', label_mode='int')
    #raw_dataset = tf.data.TFRecordDataset("/home/jovyan/mnt/external-images-pvc/ximeng/train_five_channels.tfrecords")
    parsed_image_dataset = raw_dataset.map(_parse_function)
    return parsed_image_dataset

    #raw_dataset = tf.data.TFRecordDataset(tfrecord_file_path)
    #parsed_image_dataset = dataset.map(read_tfrecord)

    # raw_dataset = tf.data.TFRecordDataset(tfrecord_file_path)

    # image_feature_description = {
    #     'label': tf.io.FixedLenFeature([], tf.int64),
    #     'image_raw': tf.io.FixedLenFeature([], tf.string),}
    
    # parsed_image_dataset =tf.io.parse_single_example(raw_dataset.map, image_feature_description)
    # return parsed_image_dataset


def build_model(parsed_image_dataset):
    
    input_layer = tf.keras.layers.Input(shape=(2160,2160,5))
    second_layer = layers.Conv2D(3, 3, 9, activation='relu')(input_layer)
    resnet50_layers = tf.keras.applications.ResNet50(include_top=True, weights='imagenet')(second_layer)
    outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(resnet50_layers)
    model = tf.keras.Model(inputs=input_layer, outputs=outputs)
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.summary()

    model.fit(parsed_image_dataset, epochs=10)
    callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]
    return model
def model_probability(model, dataset):
    probability_model = tf.keras.Sequential([model, 
    tf.keras.layers.Softmax()])

    predictions = probability_model.predict(dataset)
    predictions[10]

if __name__ == "__main__":
    main()
