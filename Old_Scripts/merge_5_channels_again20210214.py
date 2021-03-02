import os
import cv2
import numpy as np
import pandas as pd
from tifffile import imread, imwrite

##merge five channels' image by add

image_path_1 = "/home/jovyan/mnt/external-images-pvc/ximeng/FileName_ORIG_CONCAVALIN"
image_path_2 = "/home/jovyan/mnt/external-images-pvc/ximeng/FileName_ORIG_HOECHST"
image_path_3 = "/home/jovyan/mnt/external-images-pvc/ximeng/FileName_ORIG_MITO"
image_path_4 = "/home/jovyan/mnt/external-images-pvc/ximeng/FileName_ORIG_PHALLOIDIN_WGA"
image_path_5 = "/home/jovyan/mnt/external-images-pvc/ximeng/FileName_ORIG_SYTO"
target_path = "/home/jovyan/mnt/external-images-pvc/ximeng/five_channel_images/"
file_name_list =  os.listdir(image_path_1)

# for file_name in file_name_list:
#     image_1 = cv2.imread(image_path_1 +"/" + file_name,-1)
#     image_2 = cv2.imread(image_path_2 +"/" + file_name,-1)
#     image_3 = cv2.imread(image_path_3 +"/" + file_name,-1)
#     image_4 = cv2.imread(image_path_4 +"/" + file_name,-1)
#     image_5 = cv2.imread(image_path_5 +"/" + file_name,-1)


#     five_channel_image = np.zeros((image_1.shape[0], image_1.shape[1], 5))
#     five_channel_image [:,:,0] = image_1
#     five_channel_image [:,:,1] = image_2
#     five_channel_image [:,:,2] = image_3
#     five_channel_image [:,:,3] = image_4
#     five_channel_image [:,:,4] = image_5
#     print(file_name)
#     np.save(target_path + file_name,five_channel_image)


df = pd.read_csv("/home/jovyan/mnt/external-images-pvc/ximeng/dataset_ximeng.csv",';')


add_group = ['AGC']
for index, row in df.iterrows():
    if row['group'] in add_group:
        image_1 = cv2.imread(image_path_1 +"/" + str(index) + ".tif",-1)
        image_2 = cv2.imread(image_path_2 +"/" + str(index) + ".tif",-1)
        image_3 = cv2.imread(image_path_3 +"/" + str(index) + ".tif",-1)
        image_4 = cv2.imread(image_path_4 +"/" + str(index) + ".tif",-1)
        image_5 = cv2.imread(image_path_5 +"/" + str(index) + ".tif",-1)


        five_channel_image = np.zeros((image_1.shape[0], image_1.shape[1], 5))
        five_channel_image [:,:,0] = image_1
        five_channel_image [:,:,1] = image_2
        five_channel_image [:,:,2] = image_3
        five_channel_image [:,:,3] = image_4
        five_channel_image [:,:,4] = image_5
        print(index)
        np.save(target_path + str(index), five_channel_image)


#remove 'other' group
other_group = ['Atypical']
for index, row in df.iterrows():
    if row['group'] in other_group:
        print(str(index))
        os.remove('/home/jovyan/mnt/external-images-pvc/ximeng/five_channel_images/' + str(index) + '.npy')

