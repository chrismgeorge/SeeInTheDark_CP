import os, time, scipy.io, pdb

import rawpy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from GLOBALS import * 
from train import train
from model import SeeInDark
from structured_svm import SeeInDark_Structured_SVM
from SID_Dataset import Dataset
from utils import get_train_ids, get_test_ids


def main():
    print('Getting Train/Test Ids')
    train_ids = get_train_ids()
    test_ids = get_test_ids()
    
    print('Check debugging')
    DEBUG = 0
    if DEBUG == 1:
        save_freq = 100
        train_ids = train_ids[0:5]
        test_ids = test_ids[0:5]

    print('Making model and dataset')
    model = SeeInDark_Structured_SVM()
    model._initialize_weights()
    
    train_dataset = Dataset(train_ids)
    train_loader_args = dict(shuffle=True, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    dataloader = data.DataLoader(train_dataset, **train_loader_args)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print('Training...')
    train_losses = []
    
    #for epoch in range(0, 5000):
    for epoch in range(0, 5001):
        # time it
        start_time = time.time()
        
        # Run an epoch
        train_loss = train_svm(model, epoch, dataloader, optimizer)
        train_losses.append(train_loss)
        
        # time it
        end_time = time.time()
        print("Epoch time: ", end_time - start_time, "Running loss: ", train_loss)
        

main()