#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os,time,scipy.io

import numpy as np
import rawpy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from model import SeeInDark
from GLOBALS import * 
from SID_Dataset import Dataset
from utils import get_train_ids, get_test_ids
import pdb


def main():
    print('Getting Train/Test Ids')
    train_ids = get_train_ids()
    test_ids = get_test_ids()
    
    DEBUG = 0
    if DEBUG == 1:
        save_freq = 100
        train_ids = train_ids[0:5]
        test_ids = test_ids[0:5]

    # Raw data takes long time to load. Keep them in memory after loaded.
    gt_images=[None]*6000
    input_images = {}
    input_images['300'] = [None]*len(train_ids)
    input_images['250'] = [None]*len(train_ids)
    input_images['100'] = [None]*len(train_ids)

    model = SeeInDark().to(device)
    model._initialize_weights()
    
    pdb.set_trace()
    
    train_dataset = SID_Dataset(train_ids)
    train_loader_args = dict(shuffle=True, batch_size=1, num_workers=num_workers, pin_memory=True)
    dataloader = data.DataLoader(train_dataset, **train_loader_args)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    train_losses = []
    for epoch in range(0, 5000):
        start_time = time.time()
        train_loss = train(model, epoch, dataloader, optimizer)
        end_time = time.time()
        print("Epoch time: ", end_time - start_time)
        train_losses.append(train_loss)

main()


# In[ ]:




