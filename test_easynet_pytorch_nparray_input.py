import torch
import pandas as pd
import PIL
import os
import matplotlib.pyplot as plt
import time
import numpy as np
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_folder, transform = None):
        self.df = pd.read_csv(csv_path,sep = ';')
        self.images_folder = images_folder
        self.transform = transform
        self.class2index = {"compound":0, "control":1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.loc[index,'id']
        label =  self.class2index[self.df.loc[index, 'type']]
        image = np.load(os.path.join(self.images_folder, str(filename)) + '.npy')
        if self.transform is not None:
            image = self.transform(image)
        return image, label
       
       
train_dataset = CustomDataset("/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/test1_train_dataset.csv", "/home/jovyan/mnt/external-images-pvc/ximeng/five_channel_images"  )
valid_dataset = CustomDataset("/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/test1_valid_dataset.csv", "/home/jovyan/mnt/external-images-pvc/ximeng/five_channel_images"  )
test_dataset = CustomDataset("/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/test1_test_dataset.csv", "/home/jovyan/mnt/external-images-pvc/ximeng/five_channel_images"  )
train_data_size = len(train_dataset)
valid_data_size = len(valid_dataset)
test_data_size = len(test_dataset)

def main_nn():

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(5, 32, 3, 1937)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

            # x represents our data
        def forward(self, x):
            # Pass data through conv1
            x = self.conv1(x)
            # Use the rectified-linear activation function over x
            x = nn.functional.relu(x)

            x = self.conv2(x)
            x = nn.functional.relu(x)

            # Run max pooling over x
            x = nn.functional.max_pool2d(x, 2)
            # Pass data through dropout1
            x = self.dropout1(x)
            # Flatten x with start_dim=1
            x = torch.flatten(x, 1)
            # Pass data through fc1
            x = self.fc1(x)
            x = nn.functional.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)

            # Apply softmax to x
            output = nn.functional.log_softmax(x, dim=1)
            return output

    model = Net()
    print(model)


    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

    return model,loss_function,optimizer

resnet50,loss_function,optimizer = main_nn()

def train_and_valid(model, loss_function, optimizer, epochs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    best_acc = 0.0
    best_epoch = 0
 
    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
 
        model.train()
 
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
 
        for i, (inputs, labels) in enumerate(train_dataset):
            inputs  = ToTensor()(inputs)
            labels = torch.tensor(labels).to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
 
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
 
            train_loss += loss.item() * inputs.size(0)
 
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
 
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
 
            train_acc += acc.item() * inputs.size(0)
 
        with torch.no_grad():
            model.eval()
 
            for j, (inputs, labels) in enumerate(valid_dataset):
                inputs  = ToTensor()(inputs)
                labels = torch.tensor(labels).to(device)
 
                outputs = model(inputs)
 
                loss = loss_function(outputs, labels)
 
                valid_loss += loss.item() * inputs.size(0)
 
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
 
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
 
                valid_acc += acc.item() * inputs.size(0)
 
        avg_train_loss = train_loss/train_data_size
        avg_train_acc = train_acc/train_data_size
 
        avg_valid_loss = valid_loss/valid_data_size
        avg_valid_acc = valid_acc/valid_data_size
 
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
 
        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
 
        epoch_end = time.time()
 
        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
            epoch+1, avg_valid_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start
        ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
 
        torch.save(model, 'models/'+'_model_'+str(epoch+1)+'.pt')
        
    return model, history


num_epochs = 10
trained_model, history = train_and_valid(resnet50, loss_function, optimizer, num_epochs)
torch.save(history, 'models/'+"5channels_test1"+'_history.pt')
 
history = np.array(history)
plt.plot(history[:, 0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.savefig('_loss_curve.png')
plt.show()
 
plt.plot(history[:, 2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig('_accuracy_curve.png')

