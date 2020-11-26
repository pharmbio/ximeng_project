import os
import cv2
import numpy as np

##merge five channels' image by add
def main():
    image_path_1 = "/home/jovyan/scratch-shared/ximeng/FileName_ORIG_CONCAVALIN"
    image_path_2 = "/home/jovyan/scratch-shared/ximeng/FileName_ORIG_HOECHST"
    image_path_3 = "/home/jovyan/scratch-shared/ximeng/FileName_ORIG_MITO"
    image_path_4 = "/home/jovyan/scratch-shared/ximeng/FileName_ORIG_PHALLOIDIN_WGA"
    image_path_5 = "/home/jovyan/scratch-shared/ximeng/FileName_ORIG_SYTO"
    target_path = "/home/jovyan/scratch-shared/ximeng/five_channel_images/"
    file_name_list =  os.listdir("/home/jovyan/scratch-shared/ximeng/FileName_ORIG_CONCAVALIN")

    for file_name in file_name_list:
        image_1 = cv2.imread(image_path_1 +"/" + file_name,-1)
        image_2 = cv2.imread(image_path_2 +"/" + file_name,-1)
        image_3 = cv2.imread(image_path_3 +"/" + file_name,-1)
        image_4 = cv2.imread(image_path_4 +"/" + file_name,-1)
        image_5 = cv2.imread(image_path_5 +"/" + file_name,-1)

        two_channel_image = cv2.add(image_1,image_2,image_3)
        three_channel_image = cv2.add(image_4,image_5)
        five_channel_image = cv2.add(two_channel_image,three_channel_image)
        cv2.imwrite(target_path + file_name,five_channel_image)
        print(file_name)

        # five_channel_image = np.zeros((image_1.shape[0], image_1.shape[1], 5))

        # five_channel_image [:,:,0] = image_1
        # five_channel_image [:,:,1] = image_2
        # five_channel_image [:,:,2] = image_3
        # five_channel_image [:,:,3] = image_4
        # five_channel_image [:,:,4] = image_5

        # cv2.imwrite(target_path + file_name,five_channel_image)

if __name__ == "__main__":
    main()

