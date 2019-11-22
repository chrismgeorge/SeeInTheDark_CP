import os,time,scipy.io

import numpy as np
import rawpy
import glob
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
        

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

def save_current_model(model, epoch, out_img):
    # Make directory if empty
    if not os.path.isdir(result_dir + '%04d'%epoch):
        os.makedirs(result_dir + '%04d'%epoch)
    
    # Re-Permute the image
    output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
    output = np.minimum(np.maximum(output,0),1)
    temp = np.concatenate((gt_patch[0,:,:,:], output[0,:,:,:]), axis=1)
    
    # Save test out image
    scipy.misc.toimage(temp*255,  high=255, low=0, cmin=0, cmax=255).save(
                       result_dir + '%04d/%05d_00_train_%d.jpg'%(epoch,train_id,ratio)
                      )
    
    # Save model
    torch.save(model.state_dict(), model_dir+'checkpoint_sony_e%04d.pth'%epoch)


def train(model, epoch, dataloader, optimizer):
    if epoch > 2000:
        for g in opt.param_groups:
            g['lr'] = 1e-5
    
    running_loss = 0.0
    for (in_img, out_image) in dataloader:
        model.zero_grad()
        out_img = model(in_img)

        loss = reduce_mean(out_img, gt_img) # Criterion
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item 
        
        if epoch % save_freq == 0:
            save_current_model(model, epoch, out_img)
   
    return running_loss
            

