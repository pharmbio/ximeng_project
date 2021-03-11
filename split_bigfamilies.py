#creat a csv file with only big 3 family/groups iamges
import pandas as pd
import os

df = pd.read_csv("/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/essencial_csv_file.csv",';',dtype='str')
new_file_list = list()
file_list = os.listdir("/home/jovyan/mnt/external-images-pvc/ximeng/five_channel_images")
for i in file_list:
    i = i.split('.')[0]
    new_file_list.append(i)

print(len(new_file_list))
newdf = df[df['id'].isin(new_file_list)]
print(newdf)
newdf.to_csv("/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/march10_big_3_essencial_csv_file.csv",';',index=None)



#save csv file with only biggest 3 families

import pandas as pd
df = pd.read_csv('/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/march10_big_3_essencial_csv_file.csv', sep = ';',dtype='str')

#change family column empty to control
df.loc[df['type'] == "control",'family'] = 'control'
df.to_csv('/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/march2_big_3_essencial_csv_file.csv', sep=';',index=None)

#creat a new csv file with only big 3 family iamges
big_family = ['EGFR', 'PIKK', 'CDK']
big_families_data = df[(df['family'].isin(['EGFR', 'PIKK', 'CDK'])) | (df['type'] == "control")]
print(big_families_data)
big_families_data.to_csv('/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/only_big_3_families.csv')


#split train test dataset
new_df =  pd.read_csv('/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/only_big_3_families.csv')

new_df = big_families_data.sample(frac=1.0)  # shuffle
cut_idx = int(round(0.2 * new_df.shape[0]))
new_df_test, new_df_train = new_df.iloc[:cut_idx], new_df.iloc[cut_idx:]
print (new_df.shape, new_df_test.shape, new_df_train.shape)  

new_df_test.to_csv('/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/only_big_3_families_test_dataset.csv', sep=';',index=None)
new_df_train.to_csv('/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/only_big_3_families_train_dataset.csv', sep=';',index=None)