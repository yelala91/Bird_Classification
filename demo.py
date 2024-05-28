# demo.py
#
# ============================================
# A demo to tune a pretrained model Resnet-18
# ============================================

import os
import torch
import torch.optim as optim  
# from torch.utils.tensorboard import
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.resnet import resnet18
from torchvision import transforms
from datetime import datetime

import os
import sys; sys.path.append('.'+ os.sep + 'private')

from private.data_read import data_read
import private.model as md
import private.train as tr

def main():
    NUM_CLASSES     = 200
    TRAIN_TXT_PATH  = '.' + os.sep + 'convert_txt' + os.sep + 'train_set.txt'
    TEST_TXT_PATH   = '.' + os.sep + 'convert_txt' + os.sep + 'test_set.txt'
    IMAGES_PATH     = '.' + os.sep + 'dataset' + os.sep + 'CUB_200_2011' + os.sep + 'images'
    SAVE_PATH       = '.' + os.sep + 'saves'

    DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS          = 50
    BATCH_SIZE      = 64
    NUM_WORKERS     = 4
    LEARNING_RATE   = 1e-2
    MOMENTUM        = 0.9

    pretrained          = False
    fine_tuning_rate    = 1e-1

    # set model
    
    model = md.bird_resnet18
    if pretrained:
        weights = torch.load('checkpoints/resnet18-f37072fd.pth')
        del weights['fc.weight']
        del weights['fc.bias']
        model.load_state_dict(weights, strict=False)
    # model = md.init_model(pre_model="resnet18", num_classes=NUM_CLASSES, pretrained=pretrained)

    # set transform of images
    train_preprocess = transforms.Compose([  
        transforms.Resize((224, 224)),  
        # transforms.CenterCrop(224),
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    valid_preprocess = transforms.Compose([  
        transforms.Resize((224, 224)),  
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    ])
    
    folder_name = ('pretrained' if pretrained else 'unpretrained') + os.sep + datetime.now().strftime("%Y_%m_%d_%H_%M")
    train_writer = SummaryWriter(os.path.join(SAVE_PATH, folder_name + os.sep + 'run/train'))
    valid_writer = SummaryWriter(os.path.join(SAVE_PATH, folder_name + os.sep + 'run/valid'))

    # load data
    train_loader, valid_loader = data_read(TRAIN_TXT_PATH, IMAGES_PATH, valid_rate=0.1, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, transform=train_preprocess, type='train')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    test_loader = data_read(TEST_TXT_PATH, IMAGES_PATH, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, transform=valid_preprocess, type='test')
    
    model   = model.to(DEVICE)  
    
    # set oprimizer
    if pretrained:
        params = list(model.parameters())
        optimizer = optim.SGD(params[:-2], lr=fine_tuning_rate*LEARNING_RATE, momentum=MOMENTUM) #, momentum=MOMENTUM)  
        optimizer.add_param_group({'params': params[-2:], 'lr': LEARNING_RATE, 'momentum':MOMENTUM, 'weight_decay':5e-4})
    else:
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=1e-3)

    # start train and save the model
    tr.train(model, train_loader, valid_loader, DEVICE, EPOCHS, optimizer=optimizer, writers=(train_writer, valid_writer))

    torch.save(model.state_dict(), SAVE_PATH + os.sep + folder_name + os.sep + 'model.pth')

    # start test
    tr.test(model, test_loader, DEVICE)
    
    train_writer.close()
    valid_writer.close()


if __name__ == '__main__':
    main()
