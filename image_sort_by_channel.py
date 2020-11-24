import pandas as pd 
import shutil


##To move single channel images into sub-folder
def main():
    df = read_dataframe()
    sort_by_channel('FileName_ORIG_CONCAVALIN',df)
    sort_by_channel('FileName_ORIG_HOECHST',df)
    sort_by_channel('FileName_ORIG_MITO',df)
    sort_by_channel('FileName_ORIG_PHALLOIDIN_WGA',df)
    sort_by_channel('FileName_ORIG_SYTO',df)
    print('Done')

def read_dataframe():
    return pd.read_csv("~/scratch-shared/ximeng/dataset_ximeng.csv",',')

def sort_by_channel(column_name,dataframe):
    for index, row in dataframe.iterrows():
        image_name = row[column_name]
        shutil.copy("/home/jovyan/scratch-shared/ximeng/all_single_channel_images/"+ image_name, "/home/jovyan/scratch-shared/ximeng/"+ column_name +'/'+ index + '.tif')


if __name__ == "__main__":
    main()
