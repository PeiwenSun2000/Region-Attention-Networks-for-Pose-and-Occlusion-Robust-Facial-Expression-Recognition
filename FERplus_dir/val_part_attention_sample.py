import os, sys, shutil
import random as rd
from os import listdir
from PIL import Image
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
import pdb

def load_imgs_plus(img_dir, list_txt, dict_data):
    imgs_first = list()
    video_path=img_dir+list_txt
    img_lists = listdir(video_path)
    for i in range(len(img_lists)):
       img_path_first = video_path+'/'+img_lists[i]
       #pdb.set_trace()
       imgs_first.append((img_path_first,int(dict_data[list_txt][1])))
    return imgs_first

class MsCelebDataset(data.Dataset):
    def __init__(self, img_dir,list_txt,dict_data, transform=None):

        self.imgs_first = load_imgs_plus(img_dir,list_txt,dict_data)
        #pdb.set_trace()
        self.transform = transform

    def __getitem__(self, index):
        # pdb.set_trace()
        path_firt, target_first = self.imgs_first[index]
        img_first = Image.open(path_firt).convert("RGB")
        if self.transform is not None:
            img_first = self.transform(img_first)
        
        return img_first, target_first 
    
    def __len__(self):
        return len(self.imgs_first)




class CaffeCrop(object):
    """
    This class take the same behavior as sensenet
    """
    def __init__(self, phase):
        assert(phase=='train' or phase=='test')
        self.phase = phase

    def __call__(self, img):
        # pre determined parameters
        final_size = 224
        final_width = final_height = final_size
        res_img = img.resize( (final_width, final_height) )
        return res_img
