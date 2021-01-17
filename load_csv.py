import torch
import pandas as pd
import PIL
import os
import csv

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_folder, transform = None):
        self.df = pd.read_csv(csv_path,sep = ';')
        self.images_folder = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        filename = self.df[index,'id']
        label = self.df[index, 'type']
        image = PIL.Image.open(os.path.join(self.images_folder, filename,'.tif'))
        if self.transform is not None:
            image = self.transform(image)
        return image, label
       
train_dataset = CustomDataset("/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/test1_train_dataset.csv", "/home/jovyan/mnt/external-images-pvc/ximeng/all_tif_image_files_five_channels"  )
test_dataset = CustomDataset("/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/test1_test_dataset.csv", "/home/jovyan/mnt/external-images-pvc/ximeng/all_tif_image_files_five_channels"  )
valid_dataset = CustomDataset("/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/test1_valid_dataset.csv", "/home/jovyan/mnt/external-images-pvc/ximeng/all_tif_image_files_five_channels"  )

image, label = valid_dataset[3]
