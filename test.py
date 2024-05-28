import sys
import os
import torch
import private.model as md
import private.train as tr
from torchvision import transforms
from private.data_read import data_read

if __name__ == '__main__':
    DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
    TEST_TXT_PATH   = '.' + os.sep + 'convert_txt' + os.sep + 'test_set.txt'
    IMAGES_PATH     = '.' + os.sep + 'dataset' + os.sep + 'CUB_200_2011' + os.sep + 'images'
    BATCH_SIZE      = 64
    NUM_WORKERS     = 4
    
    weights_path = sys.argv[1]
    
    weights = torch.load(weights_path)
    model = md.bird_resnet18
    
    model.load_state_dict(weights)
    
    test_preprocess = transforms.Compose([  
        transforms.Resize((224, 224)),  
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    ])
    test_loader = data_read(TEST_TXT_PATH, IMAGES_PATH, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, transform=test_preprocess, type='test')
    
    tr.test(model, test_loader, device=DEVICE)