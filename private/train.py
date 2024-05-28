# model.py
#
# ==========================================
# 
# ==========================================

import torch  
import torch.nn as nn  
# from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

def train(model, train_loader, valid_loader, device, epochs, optimizer, writers=None):
    model.to(device)
    batches  = len(train_loader)
    if writers is not None:
        train_writer, valid_writer = writers
        
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = CosineAnnealingLR(optimizer, epochs*batches, eta_min=0)
    criterion = nn.CrossEntropyLoss()  
    
    for epoch in range(epochs):  
        model.train()
        running_loss = 0.0  
        correct = 0; total   = 0  
        for inputs, labels in train_loader:  
            inputs = inputs.to(device)  
            labels = labels.to(device)  
    
            optimizer.zero_grad()  
    
            outputs         = model(inputs)
            _, predicted    = torch.max(outputs.data, 1)  
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)  
            
            loss = criterion(outputs, labels)  

            loss.backward()  
            optimizer.step()  

            running_loss += loss.item()  
        scheduler.step()
        averge_loss = running_loss/batches
        train_acc = correct / total
        
        if writers is not None:
            train_writer.add_scalar('Loss', averge_loss, epoch)
            # train_writer.add_scalar('Accuracy', train_acc, epoch)
            
        print(f'Epoch {epoch+1}/{epochs}, Loss: {averge_loss}, Acc: {train_acc*100:2f}%')  

        if (epoch+1) % 1 == 0:
            model.eval() 
            with torch.no_grad():  
                correct = 0  
                total   = 0  
                running_loss = 0.0
                for images, labels in valid_loader:  
                    images  = images.to(device)  
                    labels  = labels.to(device)
                    
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
                    
                    _, predicted = torch.max(outputs.data, 1)  
                    total   += labels.size(0)  
                    correct += (predicted == labels).sum().item()
                    
                    running_loss += loss
                    
                averge_loss = running_loss / len(valid_loader)
                valid_acc   = correct / total
                if writers is not None:
                    valid_writer.add_scalar('Loss', averge_loss, epoch)
                    valid_writer.add_scalar('Accuracy', valid_acc, epoch)
                    
                print(f'Accuracy of the network on the {len(valid_loader.dataset)} test images: {100 * valid_acc: 2f}%')  

    
def test(model, test_loader, device):
    print('Start to test!')
    model.eval() 
    model.to(device)
    with torch.no_grad():  
        correct = 0  
        total   = 0  
        for images, labels in tqdm(test_loader):  
            images  = images.to(device)  
            labels  = labels.to(device)  
            outputs = model(images)  
            _, predicted = torch.max(outputs.data, 1)  
            total   += labels.size(0)  
            correct += (predicted == labels).sum().item()  
    
        print(f'Accuracy of the network on the {len(test_loader.dataset)} test images: {100 * correct / total}%')  
        
def double_test(model_1, model_2, test_loader, device):
    model_1.eval()
    model_2.eval()
    
    with torch.no_grad():
        correct = 0
        total   = 0
        for images, labels in tqdm(test_loader): 
            images  = images.to(device)  
            labels  = labels.to(device)  
            
            outputs_1 = model_1(images)
            outputs_2 = model_2(images)
            
            _, predicted = torch.max(outputs_1.data+outputs_2.data, 1)  
            
            total += labels.size(0) 
            correct += (predicted == labels).sum().item()  
        
        print(f'Accuracy of the network on the {len(test_loader.dataset)} test images: {100 * correct / total}%')