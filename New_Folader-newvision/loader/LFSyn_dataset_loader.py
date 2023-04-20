import os
import torch
import numpy as np
import imageio
import scipy.io
from torch.utils import data
import random


class LFSyndatasetLoader(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        augmentations=None,
        test_mode=False,
        model_name=None
    ):
        self.root = root        
        self.split = split    
        self.augmentations = augmentations  
        self.test_mode=test_mode   
        self.model_name=model_name 
        self.n_classes = 14 
        self.files = [] 
        if self.split == "train":
            self.files = os.listdir(self.root+'/train/')
        elif self.split == "val":
            self.files = os.listdir(self.root+'/val/')
        else:
            self.files = os.listdir(self.root+'/test/')
        print("Found %d %s images" % (len(self.files), split))

    def __len__(self):     
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):   
        """__getitem__

        :param index:
        """
        if not self.test_mode:      
            imgname = self.files[index].rstrip()    
            img_path = os.path.join(self.root,self.split,imgname,"5_5.png")
            lbl_path = os.path.join(self.root,self.split,imgname,"5_5_label.npy")
            lbl = np.load(lbl_path)-1       
            lbl = np.array(lbl,dtype=np.uint8)
            
            img = imageio.imread(img_path)
            img = np.array(img, dtype=np.uint8)
            if self.augmentations is not None:
                [img], lbl = self.augmentations([img], lbl)
            
            img = img.float()
            lbl = torch.from_numpy(lbl).long()  

            return img,lbl,img_path
