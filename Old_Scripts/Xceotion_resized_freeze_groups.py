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
import pretrainedmodels
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def main():
    torch.cuda.empty_cache()#clean cache before running
    #load data
    train_dataset = CustomDataset('/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/only_big_3_groups_train_dataset.csv', "/home/jovyan/scratch-shared/ximeng/resized_image"  )
    valid_dataset = CustomDataset('/home/jovyan/mnt/external-images-pvc/ximeng/csv_files_for_load/only_big_3_groups_test_dataset.csv', "/home/jovyan/scratch-shared/ximeng/resized_image"  )
    train_data_size = len(train_dataset)
    valid_data_size = len(valid_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, shuffle=True, num_workers=8)
    
    working_device = "cuda:0" #if torch.cuda.is_available() else "cpu"
    select_model,loss_function,optimizer = main_nn(working_device)

    num_epochs = 20
    file_save_name = '0316_groups_Xception_resize_20epoch'
    trained_model, history, filenames, class_preds, class_true= train_and_valid(working_device, select_model, loss_function, optimizer, num_epochs, train_dataloader, valid_dataloader, train_data_size,valid_data_size)
    
    save_and_plot(trained_model, history, file_save_name, filenames, class_preds, class_true)
    


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_folder, transform = None):
        self.df = pd.read_csv(csv_path,sep = ';')
        self.images_folder = images_folder
        self.transform = transform
        self.class2index ={"control":0, "TK":1, "CMGC":2, "AGC":3}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.loc[index,'id']
        label =  self.class2index[self.df.loc[index, 'group']]
        image = np.load(os.path.join(self.images_folder, str(filename)) + '.npy')
        if self.transform is not None:
            image = self.transform(image)
        return image, label, filename
       

def main_nn(working_device):
    model = pretrainedmodels.xception(num_classes=1000, pretrained='imagenet')
    
    model.last_linear.out_features = 4
    print(model)   
    model = nn.Sequential(
         nn.Conv2d(in_channels=5, 
                    out_channels=3, 
                    kernel_size=3, 
                    stride=3, 
                    padding=0,
                    bias=False),
        model)

    if torch.cuda.is_available():
        #model = torch.nn.DataParallel(model, device_ids=[0,1,2]) #multi gpu
        model.to(torch.device(working_device))

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

    return model,loss_function,optimizer


def train_and_valid(working_device, model, loss_function, optimizer, epochs, train_dataloader, valid_dataloader, train_data_size,valid_data_size):

    device = torch.device(working_device)
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
        filenames, class_preds, class_true= [], [], []
        for i, (inputs, labels,filename) in enumerate(train_dataloader):
            inputs = torch.tensor(inputs, dtype=torch.float32)
            inputs = inputs.transpose(1, 3).contiguous()
            inputs, labels = inputs.to(device), labels.to(device)

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
            for j, (inputs, labels,filename) in enumerate(valid_dataloader):
                inputs = torch.tensor(inputs, dtype=torch.float32)
                inputs = inputs.transpose(1, 3).contiguous()
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
 
                loss = loss_function(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
 
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                filenames.extend(filename.data.cpu().tolist())
                class_preds.extend(predictions.data.cpu().tolist())
                class_true.extend(labels.data.cpu().tolist())
                #print(filenames, class_preds, class_true)

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

        #torch.cuda.empty_cache()

        #print(filenames, class_preds, class_true)
    return model, history, filenames, class_preds, class_true


def save_and_plot(trained_model, history, file_save_name, filenames, class_preds, class_true):

    torch.save(trained_model, '/home/jovyan/repo/ximeng_project/Outputs/'+file_save_name+'_trained_model.pt')
        
    torch.save(history, '/home/jovyan/repo/ximeng_project/Outputs/'+file_save_name+'_history.pt') 
    history = np.array(history)
    plt.plot(history[:, :])
    plt.legend(['Train Loss', 'Valid Loss','Train Accuracy', 'Valid Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig('/home/jovyan/repo/ximeng_project/Outputs/'+ file_save_name + '_all.png')
    plt.show()

    y_pred = class_preds
    y_true = class_true
    cm = confusion_matrix(y_true, y_pred,labels=[0,1,2,3], normalize='all')
    cmplot =  ConfusionMatrixDisplay(cm,display_labels=["control", "TK", "CMGC", "AGC"])
    cmplot.plot()
    plt.savefig('/home/jovyan/repo/ximeng_project/Outputs/'+ file_save_name + '_cmplot.png')
    plt.show()
    np.savetxt( '/home/jovyan/repo/ximeng_project/Outputs/'+ file_save_name+'true_vs_preds_output.csv', [filenames, class_preds, class_true]) 


if __name__ == "__main__":
    main()
