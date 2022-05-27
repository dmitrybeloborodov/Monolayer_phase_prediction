import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

def split_train_val(dataset, val_percent=0.05):
    """Splits the dataset into training and validation portions"""
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}

def normalize(x):
    """Maps the coords to range [-1,1]"""
    return x / 200

def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))

def one_hot(x):
    """One hot encoding the label"""
    encoded = np.zeros(3)
    encoded[x] = 1.
    return encoded

def read_file(f):
    """Read a trajectory frame"""
    temp=np.zeros((10000,2))
    for j in range (10):
        line=f.readline()
        if line=="":
            return temp
        #if line=='ITEM: BOX BOUNDS pp pp pp':
        if line=='ITEM: ATOMS id type x y z \n':
            for i in range(10000):
                line=f.readline()

                line=line.split(' ')
                temp[i]=[float(line[2]),float(line[3])]
            break
                
    return temp

def full_read_file(name_fime):
    """Fully read a trajectory"""
    img=[]
    with open(name_fime,'r') as f:
        temp=1
        while True:
            temp=read_file(f)
            if temp.all()==0:
                break
            img.append(temp)
    img=np.array(img)
    #print(img.shape)
    return img

class TrajDataset(Dataset):
    """Dataset composed of full trajectories (used for conv_model)"""
    def __init__(self, annotations_file, data_dir, transform=None, target_transform=None):
        self.labels = pd.read_csv(annotations_file, header=None)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.labels.shape[1]

    def __getitem__(self, idx):
        traj_path = os.path.join(self.data_dir, self.labels.iloc[0, idx])
        traj = full_read_file(traj_path)
        label = int(self.labels.iloc[1, idx])
        if self.transform:
            traj = self.transform(traj)
        if self.target_transform:
            label = self.target_transform(label)
        return traj, label
    
class FrameDataset(Dataset):
    """Dataset composed of trajectory frames (used for lin_model)"""
    def __init__(self, annotations_file, data_dir, transform=None, target_transform=None):
        self.labels = pd.read_csv(annotations_file, header=None)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.labels.shape[1]*801

    def __getitem__(self, idx):
        traj_path = os.path.join(self.data_dir, self.labels.iloc[0, idx//801])
        traj = full_read_file(traj_path)
        to_take = idx % 801
        frame = traj[to_take].flatten()
        label = self.labels.iloc[1, idx//801]
        if self.transform:
            frame = self.transform(frame)
        if self.target_transform:
            label = self.target_transform(label)
        return frame, label