import os, scipy.io, pdb

import glob
import rawpy
import torch
import numpy as np
from GLOBALS import *

def get_train_ids():
    # Get train ids
    train_fns = glob.glob(gt_dir + '0*.ARW')
    train_ids = []
    for i in range(len(train_fns)):
        _, train_fn = os.path.split(train_fns[i])
        train_ids.append(int(train_fn[0:5]))
    return train_ids

def get_test_ids():
    # Get test ids
    test_fns = glob.glob(gt_dir + '/1*.ARW')
    test_ids = []
    for i in range(len(test_fns)):
        _, test_fn = os.path.split(test_fns[i])
        test_ids.append(int(test_fn[0:5]))
    return test_ids

def pack_raw(raw):
    # Pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32) 
    im = np.maximum(im - 512,0)/ (16383 - 512) # Subtract the black level

    im = np.expand_dims(im,axis=2) 
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out

def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()

def save_current_model(model, epoch, out_img, gt_img, train_id, ratio):
    # Make directory if empty
    if not os.path.isdir(result_dir + '%04d'%epoch):
        os.makedirs(result_dir + '%04d'%epoch)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    
    # Re-Permute the image
    output = out_img.permute(1, 2, 0).cpu().data.numpy()
    gt_patch = gt_img.permute(1, 2, 0).cpu().data.numpy()
    
    output = np.minimum(np.maximum(output, 0), 1)
    temp = np.concatenate((gt_patch[:,:,:], output[:,:,:]), axis=0)
#     pdb.set_trace()
    
    # Save test out image
    scipy.misc.toimage(temp*255,  high=255, low=0, cmin=0, cmax=255).save(
                       result_dir + '%04d/%05d_00_train_%d.jpg'%(epoch, train_id, ratio)
                      )
    
    # Save model
    torch.save(model.state_dict(), model_dir + 'checkpoint_sony_e%04d.pth' % epoch)