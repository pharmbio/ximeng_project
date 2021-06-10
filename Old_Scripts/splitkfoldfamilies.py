#creat a csv file with only big 3 family/groups iamges
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
df = pd.read_csv('/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/only_big_3_families.csv')

n_splits = 5  
x = df['id'].values
y = df['family'].values

skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

for index,(train_index,test_index) in enumerate(skf.split(x,y), start=1):
    res_train = pd.DataFrame()
    res_train['id'] = df['id'].iloc[train_index]
    res_train['family'] = df['family'].iloc[train_index]
    res_train.to_csv("/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/families_fold_train_{}.csv".format(index), sep=';',index=None)

    res_train = pd.DataFrame()
    res_train['id'] = df['id'].iloc[test_index]
    res_train['family'] = df['family'].iloc[test_index]
    res_train.to_csv("/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/families_fold_test_{}.csv".format(index), sep=';',index=None)
