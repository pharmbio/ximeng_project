import os
import numpy as np
from skimage.transform import resize
traversal_file="/home/jovyan/mnt/external-images-pvc/ximeng/five_channel_images"
output_file="/home/jovyan/scratch-shared/ximeng/resized_image"

def resize_image(img_path,save_path):
    image = np.load(img_path)
    image_processing = resize(image, (270,270,5),anti_aliasing=True)
    np.save(save_path, image_processing)
    print("Save as : " + save_path)

def show_files(path, all_files):
    file_list = os.listdir(path)
    for file in file_list:
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            show_files(cur_path, all_files)
        else:
            all_files.append(path+"/"+file)
    return all_files

if os.path.isdir(traversal_file):
    print("Check traversal file ok")
else:
    print("Traversal file error")

if os.path.isdir(output_file):
    print("Check water mask file ok")
else:
    print("Water mask file warning,auto create it")
    os.mkdir(output_file)


contents = show_files(traversal_file, [])

for content in contents:
    
    if content.endswith('npy'):
        print("processing : "+content)
        resize_image(content,output_file + "/" +os.path.basename(content))
