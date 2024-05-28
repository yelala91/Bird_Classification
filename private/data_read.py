# data_read.py
#
# ============================================================
# read and transform the data.
# ============================================================


from torch.utils.data import DataLoader 
from torch.utils.data.dataset import Dataset, random_split
from PIL import Image
import os

# set the dataset
class BirdSet(Dataset):
    def __init__(self, txt_path, images_path, transform=None):
        self.txt_path       = txt_path
        self.images_path    = images_path
        self.transform      = transform
        
        self.images = []
        self.labels = []
        self.boxes  = []
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            line = f.readline()
            while line:
                line_split = line.strip().split()
                image, label = line_split[0], line_split[-1]
                self.images.append(image)
                self.labels.append(int(label))
                self.boxes.append(tuple([int(float(x)) for x in line_split[1:-1]]))
                line = f.readline()
        self.num_classes = max(self.labels) + 1
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_path, img_name)
        img = Image.open(img_path).convert('RGB')
        x, y, w, h = self.boxes[idx]
        img = img.crop((x, y, x+w, y+h))
        
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

# convert the .txt from CUB_200_2011 to a easier file.
def txt_convert(CUB_path):
    train_test_split   = CUB_path + os.sep + 'train_test_split.txt'
    images             = CUB_path + os.sep + 'images.txt'
    image_class_labels = CUB_path + os.sep + 'image_class_labels.txt'
    bounding_boxes     = CUB_path + os.sep + 'bounding_boxes.txt'
    
    id_image        = {}
    train_or_test   = {}
    id_class        = {}
    id_bounding_box = {}
    
    with open(images, 'r', encoding='utf-8') as IP:
        line = IP.readline()
        while line:
            id, image       = line.strip().split()
            image           = image.replace('/', os.sep)
            id_image[id]    = image
            line            = IP.readline()
    
    with open(train_test_split, 'r', encoding='utf-8') as TTSP:
        line = TTSP.readline()
        while line:
            id, val             = line.strip().split()
            train_or_test[id]   = val
            line                = TTSP.readline()
            
    with open(image_class_labels, 'r', encoding='utf-8') as ICLP:
        line = ICLP.readline()
        while line:
            id, label       = line.strip().split()
            id_class[id]    = label
            line            = ICLP.readline()
            
    with open(bounding_boxes, 'r', encoding='utf-8') as BBP:
        line = BBP.readline()
        while line:
            id, *(x1, y1, x2, y2)   = line.strip().split()
            id_bounding_box[id]     = (x1, y1, x2, y2)
            line                    = BBP.readline()
            
    train_set = 'convert_txt' + os.sep + 'train_set.txt'
    test_set  = 'convert_txt' + os.sep + 'test_set.txt'
    
    if os.path.exists(train_set):
        os.remove(train_set)
    if os.path.exists(test_set):
        os.remove(test_set)
    
    # print(os.getcwd())
    train_set = open(train_set,'a',encoding='utf-8')
    test_set  = open(test_set,'a',encoding='utf-8')
    
    for id, val in train_or_test.items():
        if val == '0':
            test_set.write(f'{id_image[id]} {id_bounding_box[id][0]} {id_bounding_box[id][1]} {id_bounding_box[id][2]} {id_bounding_box[id][3]} {int(id_class[id])-1}\n')
        elif val == '1':
            train_set.write(f'{id_image[id]} {id_bounding_box[id][0]} {id_bounding_box[id][1]} {id_bounding_box[id][2]} {id_bounding_box[id][3]} {int(id_class[id])-1}\n')
    
    train_set.close()
    test_set.close()

# read the data 
def data_read(txt_path, images_path, valid_rate=0.1, batch_size=32, num_workers=1, transform=None, type='train'):
    if type == 'train':
        birdset      = BirdSet(txt_path, images_path, transform)
        total_length = len(birdset)
    
        valid_size = int(valid_rate * total_length)
        train_size = total_length - valid_size
        
        train_set, valid_set = random_split(birdset, [train_size, valid_size])
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers) #, num_workers=8)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=num_workers) #, num_workers=8)
        
        return train_loader, valid_loader
    
    elif type == 'test':
        test_set    = BirdSet(txt_path, images_path, transform)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)# , num_workers=8)
        
        return test_loader

# run it to generate the correct convert_txt.txt for your system(Linux or Windows or others)
if __name__ == '__main__':
    CUB_path = '.' + os.sep + 'dataset' + os.sep + 'CUB_200_2011'
    txt_convert(CUB_path)
    print(f'Have changed the \'\\\' or \'/\' to {os.sep}')
    
    # txt_path = '.' + os.sep + 'convert_txt' + os.sep + 'train_set.txt'
    # images_path = '.' + os.sep + 'dataset' + os.sep + 'CUB_200_2011' + os.sep + 'images'
    
    # birdset = BirdSet(txt_path, images_path)
    # train_loader = DataLoader(birdset, batch_size=32, shuffle=True)
    
    # print(len(train_loader))