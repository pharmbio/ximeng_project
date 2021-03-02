#save csv file with only biggest 3 families

import pandas as pd
df = pd.read_csv('/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/new_big_3_essencial_csv_file.csv', sep = ';',dtype='str')

#change family column empty to control
#df.loc[df['type'] == "control",'family'] = 'control'
#df.to_csv('/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/new_big_3_essencial_csv_file.csv', sep=';',index=None)
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