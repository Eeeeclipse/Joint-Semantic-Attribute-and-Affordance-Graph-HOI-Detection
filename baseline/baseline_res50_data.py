import os
import scipy.io as sio
import numpy as np  # Missing import
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
import json

# Dataset Definition
dataset_path = './hico_20160224_det'


with open(os.path.join(dataset_path, "safe_test.json"), 'r') as file:
    safe_test = json.load(file)

with open(os.path.join(dataset_path, "safe_train.json"), 'r') as file:
    safe_train = json.load(file)

with open(os.path.join(dataset_path, "test.json"), 'r') as file:
    test_anno = json.load(file)

with open(os.path.join(dataset_path, "train.json"), 'r') as file:
    train_anno = json.load(file)


transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
class HICODataset(Dataset):
    def __init__(self, root, annotations, image_list, transform=None):
        """
        Args:
            root (str): Path to the folder containing images.
            annotations (dict): Dictionary loaded from the `anno.mat` file.
            image_list (list): List of image filenames (train or test).
            transform (callable, optional): Optional transform to apply to images.
        """
        self.root = root
        self.annotations = annotations
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # Extract image name from the image list
        image_name = self.image_list[idx]

        # Join the image name with the root directory
        img_path = os.path.join(self.root, image_name)
        
        # Load the image
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        anno = self.annotations[image_name]
        action_labels = list(anno.keys())
        label = torch.tensor(int(np.random.choice(action_labels, 1)[0]) - 1)
        label_oh = F.one_hot(label, num_classes=600)
        return img, label_oh



def load_datasets(mode, batch_size):
    if mode == 'train':
        train_dataset = HICODataset(
            root='hico_20160224_det/images/train2015',
            annotations=train_anno,
            image_list=safe_train,
            transform=transform
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)

        return train_loader
    elif mode == 'test':
        test_dataset = HICODataset(
            root='hico_20160224_det/images/test2015',
            annotations=test_anno,
            image_list=safe_test,
            transform=transform
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,num_workers=8,pin_memory=True)

        return test_loader
    else:
        assert False, 'Mode Error!!!'

def collate_fn(batch):
    return torch.stack([x[0] for x in batch]), torch.stack([x[1] for x in batch]).to(torch.float32)