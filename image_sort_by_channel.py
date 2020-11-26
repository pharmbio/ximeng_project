import pandas as pd 
import shutil
import os

##Move single channel images into sub-folder
def main():
    #move_to_all()
    df = read_dataframe()
    sort_by_channel('FileName_ORIG_CONCAVALIN',df)
    sort_by_channel('FileName_ORIG_HOECHST',df)
    sort_by_channel('FileName_ORIG_MITO',df)
    sort_by_channel('FileName_ORIG_PHALLOIDIN_WGA',df)
    sort_by_channel('FileName_ORIG_SYTO',df)
    print('Done')

##move all image into one folder, only need once
def move_to_all():
    sourceDir="/home/jovyan/scratch-shared/ximeng/PolinaG-U2OS"
    targetDir="/home/jovyan/scratch-shared/ximeng/all_single_channel_images"
    for root, dirs, files in os.walk(sourceDir):
        for file in files:
            shutil.copy(os.path.join(root,file),targetDir) 

def read_dataframe():
    return pd.read_csv("~/scratch-shared/ximeng/dataset_ximeng.csv",';')

def sort_by_channel(column_name,dataframe):
    for index, row in dataframe.iterrows():
        image_name = row[column_name]
        shutil.copyfile("/home/jovyan/scratch-shared/ximeng/all_single_channel_images/"+ image_name, "/home/jovyan/scratch-shared/ximeng/"+ column_name +'/'+ str(index) + '.tif')


if __name__ == "__main__":
    main()
