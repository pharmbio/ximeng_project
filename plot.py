import torch
import numpy as np
import matplotlib.pyplot as plt

history = torch.load( '/home/jovyan/repo/ximeng_project/Outputs/0304_biggroups_resnet18_epoch40_history.pt') 
history = np.array(history)
print(history)
plt.plot(history[:, :])
plt.legend(['Train Loss', 'Valid Loss','Train Accuracy', 'Valid Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.savefig('/home/jovyan/repo/ximeng_project/Outputs/0304_biggroups_resnet18_epoch40_history_all.png')
plt.show()