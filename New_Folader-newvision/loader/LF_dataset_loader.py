import os
import torch
import numpy as np
import imageio
import scipy.io
from torch.utils import data
import random
from  glob import glob
import fnmatch
from PIL import Image
import re
import random

class LFdatasetLoader(data.Dataset):
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
        # self.files = [] 
        self.files = []
        if self.split in ['train','val','test']: #
            self.image_dir = os.path.join(self.root,  self.split)
            self.label_dir = os.path.join(self.root, self.split)
        for img_class in os.listdir(self.image_dir):
            img_all =[]
            class_name = os.path.join(self.image_dir,img_class)
            class_name = class_name.replace('\\','/')
            if(os.path.isdir(class_name)):
                img_all_temp = glob(class_name+'/*.png')
                for item in img_all_temp:
                    item = item.replace('\\','/')
                    # print(item)
                    img_all.append(item)
                img_all.remove(class_name+'/label.png')
                oneimg_dict={}
                oneimg_dict["image"] = img_all
                oneimg_dict["label"] = glob(class_name+'/*.npy')[0]
                self.files.append(oneimg_dict)
        # if self.split == "train":
        #     trainfile = self.root + "/LF_train.txt"
        #     with open(trainfile) as f:
        #         content = f.readlines()
        #         for x in content:
        #             x=x.strip()
        #             self.files.append(x)
        # elif self.split == "val":
        #     trainfile = self.root + "/LF_val.txt"
        #     with open(trainfile) as f:
        #         content = f.readlines()
        #         for x in content:
        #             x=x.strip()
        #             self.files.append(x)
        # else:
        #     trainfile = self.root + "/LF_test.txt"
        #     with open(trainfile) as f:
        #         content = f.readlines()
        #         for x in content:
        #             x=x.strip()
        #             self.files.append(x)
        # print("Found %d %s images" % (len(self.files), split))

    def __len__(self):     
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):  
        """__getitem__

        :param index:
        """
        # if not self.test_mode:      
        #     imgname = self.files[index].rstrip()    
        #     img_path = os.path.join(self.root,imgname,"5_5.png") 
            
        #     lbl_path = os.path.join(self.root,imgname,"label.npy")
        #     lbl = np.load(lbl_path)-1       
        #     lbl = np.array(lbl,dtype=np.uint8)
            
        #     img = imageio.imread(img_path)
        #     img = np.array(img, dtype=np.uint8)
        #     if self.augmentations is not None:
        #         [img], lbl = self.augmentations([img], lbl)
            
        #     img = img.float()
        #     lbl = torch.from_numpy(lbl).long()  
        img_all=[]
        data_dict = self.files[index]
        img_list = data_dict["image"]
        label_path = data_dict["label"]
        random_number = random.randint(0, 7)
        if random_number == 0:
            for img_name in img_list:
                img_newname  =  img_name.split('/')[-1]
                if(img_newname=="5_5.png"):
                    img_all.insert(0, imageio.imread(img_name))
                    list= img_name.split('/').pop()
                    dir= "/".join(list) + '/' + 'disparity_OAVC.npy'
                    disp = np.load(dir)
                if(int(img_newname.split(".")[0][0])==int(img_newname.split(".")[0][-1]) and int(img_newname.split(".")[0][0]) <= 4):
                    img_all.append(imageio.imread(img_name))
        elif random_number == 1:
            for img_name in img_list:
                img_newname  =  img_name.split('/')[-1]
                if(img_newname=="5_5.png"):
                    img_all.insert(0, imageio.imread(img_name))
                    list= img_name.split('/').pop()
                    dir= "/".join(list) + '/' + 'disparity_OAVC.npy'
                    disp = np.load(dir)
                if(int(img_newname.split(".")[0][-1]) == 5 and int(img_newname.split(".")[0][0]) <= 4):
                    img_all.append(imageio.imread(img_name))
        elif random_number == 2:
            for img_name in img_list:
                img_newname  =  img_name.split('/')[-1]
                if(img_newname=="5_5.png"):
                    img_all.insert(0, imageio.imread(img_name))
                    list= img_name.split('/').pop()
                    dir= "/".join(list) + '/' + 'disparity_OAVC.npy'
                    disp = np.load(dir)
                if(int(img_newname.split(".")[0][0])+int(img_newname.split(".")[0][-1])==10 and int(img_newname.split(".")[0][0]) <= 4):
                    img_all.append(imageio.imread(img_name))

        elif random_number == 3:
            for img_name in img_list:
                img_newname  =  img_name.split('/')[-1]
                if(img_newname=="5_5.png"):
                    img_all.insert(0, imageio.imread(img_name))
                    list= img_name.split('/').pop()
                    dir= "/".join(list) + '/' + 'disparity_OAVC.npy'
                    disp = np.load(dir)
                if(int(img_newname.split(".")[0][0])==5 and int(img_newname.split(".")[0][-1]) >= 6):
                    img_all.append(imageio.imread(img_name))
        
        elif random_number == 4:
            for img_name in img_list:
                img_newname  =  img_name.split('/')[-1]
                if(img_newname=="5_5.png"):
                    img_all.insert(0, imageio.imread(img_name))
                    list= img_name.split('/').pop()
                    dir= "/".join(list) + '/' + 'disparity_OAVC.npy'
                    disp = np.load(dir)
                if(int(img_newname.split(".")[0][0])==int(img_newname.split(".")[0][-1]) and int(img_newname.split(".")[0][0]) >= 6):
                    img_all.append(imageio.imread(img_name))
        
        elif random_number == 5:
            for img_name in img_list:
                img_newname  =  img_name.split('/')[-1]
                if(img_newname=="5_5.png"):
                    img_all.insert(0, imageio.imread(img_name))
                    list= img_name.split('/').pop()
                    dir= "/".join(list) + '/' + 'disparity_OAVC.npy'
                    disp = np.load(dir)
                if(int(img_newname.split(".")[0][-1]) == 5 and int(img_newname.split(".")[0][0]) >=6):
                    img_all.append(imageio.imread(img_name))
        elif random_number == 6:
            for img_name in img_list:
                img_newname  =  img_name.split('/')[-1]
                if(img_newname=="5_5.png"):
                    img_all.insert(0, imageio.imread(img_name))
                    list= img_name.split('/').pop()
                    dir= "/".join(list) + '/' + 'disparity_OAVC.npy'
                    disp = np.load(dir)
                if(int(img_newname.split(".")[0][0])+int(img_newname.split(".")[0][-1])==10 and int(img_newname.split(".")[0][0]) >= 6):
                    img_all.append(imageio.imread(img_name))

        elif random_number == 7:
            for img_name in img_list:
                img_newname  =  img_name.split('/')[-1]
                if(img_newname=="5_5.png"):
                    img_all.insert(0, imageio.imread(img_name))
                    list= img_name.split('/').pop()
                    dir= "/".join(list) + '/' + 'disparity_OAVC.npy'
                    disp = np.load(dir)
                if(int(img_newname.split(".")[0][0])==5 and int(img_newname.split(".")[0][-1]) <= 4):
                    img_all.append(imageio.imread(img_name))
 
        # img = img_all[0][0]
        label = np.load(label_path)-1
        label= np.array(label,dtype=np.uint8)
        img_id = int(re.findall('\d+',img_name.split("/")[-2])[0])
        # #导入npy文件路径位置
        # disp = np.load(r'F:\BaiduNetdiskDownload\semantic_segmentation\UrbanLF-Real\train\Image3\disparity_OAVC.npy') 
        # # 维度(432, 623)
        if self.augmentations is not None:
            img_all, label, disp = self.augmentations(img_all, label, disp)
        for i in img_all:
            i = i.float()
        label = torch.from_numpy(label).long() 
        return img_all, label,img_id, disp
            
            # return img,lbl,img_path

