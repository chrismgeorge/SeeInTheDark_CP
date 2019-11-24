import time, pdb

from GLOBALS import device
from utils import pack_raw, reduce_mean, save_current_model

def train(model, epoch, dataloader, optimizer):
    model.train()
    model.to(device)
    
    if epoch > 2000:
        for g in optimizer.param_groups:
            g['lr'] = 1e-5
    
    running_loss = 0.0
    for (in_img, gt_img) in dataloader:
        # Zero gradients
        optimizer.zero_grad()
        
        # Get model outputs
        out_img = model(in_img)
        
        # Calculate loss
        loss = reduce_mean(out_img, gt_img)
        running_loss += loss.item 
        
        # Compute gradients and take step
        loss.backward()
        optimizer.step()

        if epoch % save_freq == 0:
            save_current_model(model, epoch, out_img)
   
    return running_loss
            

