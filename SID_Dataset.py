import os, time, scipy.io, pdb

import glob
import rawpy
import numpy as np

import torch
import torch.nn as nn

from GLOBALS import *
from utils import pack_raw
import torch.optim as optim
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, train_ids):
        # Ids
        self.train_ids = train_ids
        
        # Image data
        self.gt_images=[None]*6000
        self.input_images = {}
        self.input_images['300'] = [None]*len(train_ids)
        self.input_images['250'] = [None]*len(train_ids)
        self.input_images['100'] = [None]*len(train_ids)
        
        self.pre_process()
        
    def pre_process(self):
        pass
        
    def __len__(self):
#         return 1 # for testing
        return len(self.train_ids)
    
    def __getitem__(self, ind):
        # Get the path from image id
        train_id = self.train_ids[ind]
        train_id = 2
#         train_id = 00001_00_10s.ARW
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
#         print(in_files)
        
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        _, in_fn = os.path.split(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
        gt_path = gt_files[0]
#         print(in_path, gt_path)
        _, gt_fn = os.path.split(gt_path)
        in_exposure =  float(in_fn[9:-5])
        gt_exposure =  float(gt_fn[9:-5])
        ratio = min(gt_exposure/in_exposure,300)

        # Read raw image
        if self.input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            self.input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw),axis=0) *ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            self.gt_images[ind] = np.expand_dims(np.float32(im/65535.0),axis = 0)
            
        # Crop
        H = self.input_images[str(ratio)[0:3]][ind].shape[1]
        W = self.input_images[str(ratio)[0:3]][ind].shape[2]

        xx = W // 4
        yy = H // 2
        input_patch = self.input_images[str(ratio)[0:3]][ind][:,yy:yy+ps,xx:xx+ps,:]
        gt_patch = self.gt_images[ind][:, yy*2:yy*2+ps*2, xx*2:xx*2+ps*2, :]
        
        # Random flip or transpose
        if np.random.randint(2, size=1)[0] == 1:  
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1: 
            input_patch = np.flip(input_patch, axis=0)
            gt_patch = np.flip(gt_patch, axis=0)
        if np.random.randint(2, size=1)[0] == 1: 
            input_patch = np.transpose(input_patch, (0,2,1,3))
            gt_patch = np.transpose(gt_patch, (0,2,1,3))
        
        # Clip the images
        input_patch = np.minimum(input_patch,1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        
        # Place onto device and torch it
        in_img = torch.from_numpy(input_patch).permute(0,3,1,2)
        gt_img = torch.from_numpy(gt_patch).permute(0,3,1,2)
        
        return in_img, gt_img, train_id, ratio
