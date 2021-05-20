#creat a csv file with only big 3 family/groups iamges
# import pandas as pd
# import os

# df = pd.read_csv("/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/essencial_csv_file.csv",';',dtype='str')
# new_file_list = list()
# file_list = os.listdir("/home/jovyan/mnt/external-images-pvc/ximeng/five_channel_images")
# for i in file_list:
#     i = i.split('.')[0]
#     new_file_list.append(i)

# print(len(new_file_list))
# newdf = df[df['id'].isin(new_file_list)]
# print(newdf)
# newdf.to_csv("/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/march2_big_3_essencial_csv_file.csv",';',index=None)
 


#save csv file with only biggest 3 groups

import pandas as pd
df = pd.read_csv('/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/AfterQC_big_3_essencial_csv_file.csv', sep = ';',dtype='str')

# #change family column empty to control
# df.loc[df['type'] == "control",'family'] = 'control'
# df.to_csv('/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/march2_big_3_essencial_csv_file.csv', sep=';',index=None)

#creat a new csv file with only big 3 groups iamges
big_group = ['TK', 'CMGC', 'AGC']
big_groups_data = df[(df['group'].isin(['TK', 'CMGC', 'AGC'])) | (df['type'] == "control")]
print(big_groups_data)
big_groups_data.to_csv('/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/only_big_3_groups.csv')


#split train test dataset

big_groups_data = big_groups_data.sample(frac=1.0)  # shuffle
cut_idx = int(round(0.2 * big_groups_data.shape[0]))
df_test, df_train = big_groups_data.iloc[:cut_idx], big_groups_data.iloc[cut_idx:]
print (big_groups_data.shape, df_test.shape, df_train.shape)  

df_test.to_csv('/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/only_big_3_groups_test_dataset.csv', sep=';',index=None)
df_train.to_csv('/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/only_big_3_groups_train_dataset.csv', sep=';',index=None)