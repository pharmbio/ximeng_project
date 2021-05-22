import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

history = torch.load( '/home/jovyan/repo/ximeng_project/Outputs/0519_final_Resnet50_groups_20epoch_history.pt') 
history = np.array(history)
print(history)

#plt.legend(['Train Loss', 'Valid Loss','Train Accuracy', 'Valid Accuracy'])
plt.style.use('ggplot')
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot()
plt.plot(history[:, :2])
plt.legend(['Train Loss', 'Valid Loss'], loc=2)
plt.xlabel('Epoch Number')             
ax1.set_ylabel('Loss') 
#ax1.set_ylim(0,10) 
                              

ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy') 
ax2.set_ylim(0,1) 
plt.plot(history[:, 2:3], 'g')  
plt.plot(history[:, 3:4], 'y')  
plt.legend(['Train Accuracy', 'Valid Accuracy'], loc=1)    

plt.xlabel('Epoch Number')




# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(history[:, :2], '-', label = ['Train Loss', 'Valid Loss'])

# ax2 = ax.twinx()
# ax2.plot(history[:, 2:4], '-', label = ['Train Accuracy', 'Valid Accuracy'])
# ax.legend(loc=0)
# ax.grid()
# ax.set_xlabel('Epoch Number')
# ax.set_ylabel('Loss')
# ax2.set_ylabel('Accuracy')

# ax2.legend(loc=0)

#plt.savefig('/home/jovyan/repo/ximeng_project/Outputs/0330_groups_Resnet50__augmentation_resize_20epoch_history_all.png')
#plt.show()