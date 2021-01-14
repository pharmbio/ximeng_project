import matplotlib.pyplot as plt
import time
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder




def main():
    main_dataset,resnet50,loss_function,optimizer = main_nn()
    trained_model, history = train_and_valid(main_dataset, resnet50, loss_function, optimizer, epochs=10)
    torch.save(history, 'models/'+'_history.pt')
    

def main_nn():
    main_dataset = ImageFolder('/home/jovyan/mnt/external-images-pvc/ximeng/five_channel_images/')
    
    resnet50 = models.resnet50(pretrained=True)
        
    fc_inputs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Conv2d(5,3,1937),
        nn.Linear(fc_inputs, 2),
        nn.LogSoftmax(dim=1))

    resnet50 = resnet50.to('cuda:0')
    
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(resnet50.parameters())

    return main_dataset,resnet50,loss_function,optimizer

def train_and_valid(dataset, model, loss_function, optimizer, epochs=10):
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
 
        for i, (inputs, labels) in enumerate(dataset):
            inputs = inputs.to(device)
            labels = labels.to(device)
 
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
            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
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
 
        torch.save(model, 'models/'+dataset+'_model_'+str(epoch+1)+'.pt')
    return model, history

    
def plot_loss_and_accuracy():
    
    history = np.array(history)
    plt.plot(history[:, 0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig(dataset+'_loss_curve.png')
    plt.show()
    
    plt.plot(history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(dataset+'_accuracy_curve.png')
    plt.show()