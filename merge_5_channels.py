import os
from PIL.Image import Image
import cv2
import numpy as np
import pandas as pd

##merge five channels' image by add
def main():
    # image_path_1 = "/home/jovyan/mnt/external-images-pvc/ximeng/FileName_ORIG_CONCAVALIN"
    # image_path_2 = "/home/jovyan/mnt/external-images-pvc/ximeng/FileName_ORIG_HOECHST"
    # image_path_3 = "/home/jovyan/mnt/external-images-pvc/ximeng/FileName_ORIG_MITO"
    # image_path_4 = "/home/jovyan/mnt/external-images-pvc/ximeng/FileName_ORIG_PHALLOIDIN_WGA"
    # image_path_5 = "/home/jovyan/mnt/external-images-pvc/ximeng/FileName_ORIG_SYTO"
    # target_path = "/home/jovyan/mnt/external-images-pvc/ximeng/five_channel_images/"
    # file_name_list =  os.listdir(image_path_1)

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


    # #should only merge biggest 3 families and groups images, so delete rest now, will fix next time
    # df = pd.read_csv("~/scratch-shared/ximeng/dataset_ximeng.csv",';')

    # uni_family = df.drop_duplicates('compoundname')['family'].value_counts().index[0:3]
    # big_family = list(uni_family)
    # print(big_family)
    # uni_group = df.drop_duplicates('compoundname')['group'].value_counts().index[0:3]
    # big_group = list(uni_group)

    # for index, row in df.iterrows():
    #     if row['group'] not in big_group and row['family'] not in big_family:
    #         os.remove('/home/jovyan/mnt/external-images-pvc/ximeng/five_channel_images/' + str(index) + '.tif.npy')

    #should transfer images to png format before
    target_path = "/home/jovyan/mnt/external-images-pvc/ximeng/five_channel_images/"
    for files in os.listdir(target_path + "compound"):
        img = target_path + "compound/" + files
        print(img)
        data = Image.fromarray(img) 
        data.save(target_path + "compound/" + str(files[:-7]) + 'png') 

if __name__ == "__main__":
    main()

