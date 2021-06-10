#delete files doesn't pass quality control
import pandas as pd
import os
from itertools import chain

file_list = pd.read_csv('/home/jovyan/repo/ximeng_project/quality_control.csv')
file_list = file_list.values.tolist()
file_list = list(chain.from_iterable(file_list))
print(file_list)

for i in file_list:
    remove_file = "/home/jovyan/scratch-shared/ximeng/resized_image/"+ str(i) +'.npy'
    if os.path.exists(remove_file):    
        os.remove(remove_file)
    print(i)
