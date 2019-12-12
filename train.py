import time, pdb, os
import numpy as np
from GLOBALS import device, ps, BATCH_SIZE, save_freq
from utils import pack_raw, reduce_mean, save_current_model
import torch
import os, scipy.io, pdb

import glob
import rawpy
import torch
import numpy as np
from GLOBALS import *

def train(model, epoch, dataloader, optimizer):
    model.train()
    model.to(device)
    
    if epoch > 2000:
        for g in optimizer.param_groups:
            g['lr'] = 1e-5
    
    running_loss = 0.0
    count = 0
    for (in_img, gt_img, train_ids, ratios) in dataloader:
        # Twice as big because model upsamples.
        gt_img = gt_img.view((BATCH_SIZE, 3, ps*2, ps*2)).to(device).cpu().float()
        in_img = in_img.view((BATCH_SIZE, 4, ps, ps)).to(device).float()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Get model outputs
        out_img = model(in_img)
        
        # Calculate loss
        loss = reduce_mean(out_img, gt_img)
        running_loss += loss.item()
        
        # Compute gradients and take step
        loss.backward()
        optimizer.step()
        
        if count % save_freq == 0 and count == 0:
            save_current_model(model, epoch, out_img[0], gt_img[0], train_ids[0].item(), ratios[0].item())
            count += 1

    return running_loss


import matplotlib as mpl
from ortools.linear_solver import pywraplp
def loss_augmented_map_inference(deteched_unary_potentials, detached_right, detached_down, real_stuff=None, is_real=True):
    # Solver
    solver = pywraplp.Solver('LinearExample', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    objective = 0.0

    # Contraints & Objectives
    x_vars = {}
    for row in range(ps*2):
        for col in range(ps*2):
            for c in range(0, 3):
                x_vars[row, col, c] = solver.NumVar(0, 256, 'x[%d,%d,%d]' % (row, col, c))
                objective += deteched_unary_potentials[0, c, row, col] * x_vars[row, col, c]
    
    # If an edge exists between two pixels, their values are below some threshold.
    y_s = {}
    THRESH = 55
    for row in range(ps*2):
        for col in range(ps*2):
            # detached_right = 3 x 63 x 64
            # detached_down = 3 x 64 x 63
            for c in range(0, 3):
                if (row == ps*2 - 1 and col == ps*2 - 1):
                    continue
                if col == ps*2 - 1:
                    # Contrainsts
                    y_s[row, col, c, 1] = solver.NumVar(0, 1, 'y[%d,%d,%d, last_col]' % (row, col, c))
                    solver.Add(y_s[row, col, c, 1] == x_vars[row, col, c] - x_vars[row+1, col, c] - THRESH)
#                     solver.Add(y_s[row, col, c, 1] <= - x_vars[row, col, c] + x_vars[row+1, col, c] - THRESH)
                    # Objective
                    objective += detached_down[c, row, col-1] * y_s[row, col, c, 1]
                elif row == ps*2 - 1:
                    # Contrainsts
                    y_s[row, col, c, 0] = solver.NumVar(0, 1, 'y[%d,%d,%d, last_row]' % (row, col, c))
                    solver.Add(y_s[row, col, c, 0] == x_vars[row, col, c] - x_vars[row, col+1, c] - THRESH)
#                     solver.Add(y_s[row, col, c, 0] <= - x_vars[row, col, c] + x_vars[row, col+1, c] - THRESH)
                    # Objective
                    objective += detached_right[c, row-1, col] * y_s[row, col, c, 0]
                else:
                    # Contrainsts
                    y_s[row, col, c, 0] = solver.NumVar(0, 1, 'y[%d,%d,%d, last_row]' % (row, col, c))
                    solver.Add(y_s[row, col, c, 0] == (x_vars[row, col, c] - x_vars[row, col+1, c]) - THRESH)
#                     solver.Add(y_s[row, col, c, 0] <= (- x_vars[row, col, c] + x_vars[row, col+1, c]) - THRESH)
                    y_s[row, col, c, 1] = solver.NumVar(0, 1, 'y[%d,%d,%d, last_col]' % (row, col, c))
                    solver.Add(y_s[row, col, c, 1] == x_vars[row, col, c] - x_vars[row+1, col, c] - THRESH)
#                     solver.Add(y_s[row, col, c, 1] <= - x_vars[row, col, c] + x_vars[row+1, col, c] - THRESH)
                    # Objective
                    objective += detached_right[c, row, col] * y_s[row, col, c, 0]
                    objective += detached_down[c, row, col] * y_s[row, col, c, 1]
                    
    z = 1 / ((ps*2) * (ps*2) * 3)
    if (is_real):
        for row in range(ps*2):
            for col in range(ps*2):
                for c in range(3):
#                     print(row, col, c)
                    x_star = real_stuff[0, c, row, col]
                    for c in range(0, 2):
                        objective += z * (x_star * (256 - x_vars[row, col, c]) + (256 - x_star) * x_vars[row, col, c])

    # Compute sovler max and answer!
    solver.Maximize(objective)
    result = solver.Solve()
#     assert result == pywraplp.Solver.OPTIMAL
#     assert solver.VerifySolution(1e-7, True)
    
    # Format final inputs
    final_assignment_x = np.zeros((ps*2, ps*2, 3))
    for i in range(ps*2):
        for j in range(ps*2):
            for c in range(3):
                final_assignment_x[i, j, c] = x_vars[i,j,c].solution_value()
                
    final_assignment_y = np.zeros((ps*2, ps*2, 3, 2))
    for i in range(ps*2):
        for j in range(ps*2):
            for c in range(3):
                if (row == ps*2-1 and col == ps*2-1):
                    continue
                if col == ps*2-1:
                    final_assignment_y[i, j, c, 1] = y_s[i, j, c, 1].solution_value()
                elif row == ps*2-1:
                    final_assignment_y[i, j, c, 0] = y_s[i,j,c, 0].solution_value()
                else:
                    final_assignment_y[i, j, c, 0] = y_s[i,j,c, 0].solution_value()
                    final_assignment_y[i, j, c, 1] = y_s[i,j,c, 1].solution_value()
                    
    return final_assignment_x, final_assignment_y

def compute_struct_svm_loss(predicted, true, pixel_pots, detached_right, detached_down, final_assignment_y):
    loss = 0.0
    z = 1 / (ps*2 * ps*2 * 3)
    loss_term = 0
    predicted_term = 0
    true_term = 0
    for row in range(ps*2):
        for col in range(ps*2):
            for c in range(0, 3):
#                 print(row,col, c, pixel_pots.shape, predicted.shape)
                predicted_term += pixel_pots[0, c, row, col] * predicted[row, col, c]
                true_val = 1 if true[0, c, row, col] == c else 0
                true_term += pixel_pots[0, c, row, col] * true_val
                mult_term = 1 if predicted[row, col, c] == true_val else 0
                loss_term += (z * mult_term)
                if (row == ps*2 - 1 and col == ps*2 - 1):
                    continue
                if col == ps*2 - 1:
                    predicted_term += final_assignment_y[row, col, c, 1] * detached_down[c, row, col-1]
                    true_val = 1 if true_val == true[0, c, row+1, col] else 0
                    true_term += true_val * detached_down[c, row, col-1]
                elif row == ps*2 - 1:
                    predicted_term += final_assignment_y[row, col, c, 0] * detached_right[c, row-1, col]
                    true_val = 1 if true_val == true[0, c, row, col+1] else 0
                    true_term += true_val * detached_right[c, row-1, col]
                else:
                    predicted_term += final_assignment_y[row, col, c, 0] * detached_right[c, row, col]
                    predicted_term += final_assignment_y[row, col, c, 1] * detached_down[c, row, col]
                    true_val = 1 if true_val == true[0, c, row, col+1] else 0
                    true_term += true_val * detached_right[c, row, col]
                    true_val = 1 if true_val == true[0, c, row+1, col] else 0
                    true_term += true_val * detached_down[c, row, col]
                    
    loss = torch.clamp(loss_term + predicted_term + true_term, min=0)
    return loss

def save_current_model_svm(model, epoch, out_img, gt_img, train_id, ratio):
    # Make directory if empty
    if not os.path.isdir(result_dir + '%04d'%epoch):
        os.makedirs(result_dir + '%04d'%epoch)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    
    # Re-Permute the image
#     pdb.set_trace()
    output = out_img.cpu().data.numpy()
    gt_patch = gt_img.permute(1, 2, 0).cpu().data.numpy()
#     pdb.set_trace()
#     pdb.set_trace()
    output = np.minimum(np.maximum(output, 0), 1)
    temp = np.concatenate((gt_patch[:,:,:], output[:,:,:]), axis=0)
#     pdb.set_trace()
#     pdb.set_trace()
    # Save test out image
    scipy.misc.toimage(temp*255,  high=255, low=0, cmin=0, cmax=255).save(
                       result_dir + '%04d/%05d_00_train_%d.jpg'%(epoch, train_id, ratio)
                      )
    
    # Save model
#     torch.save(model.state_dict(), model_dir + 'checkpoint_sony_e%04d.pth' % epoch)


def train_svm(model, epoch, dataloader, optimizer):
    model.train()
    model.to(device)
    
    if epoch > 2000:
        for g in optimizer.param_groups:
            g['lr'] = 1e-5
    
    running_loss = 0.0
    count = 0
    for (in_img, gt_img, train_ids, ratios) in dataloader:
        # Twice as big because model upsamples.
        gt_img = gt_img.view((BATCH_SIZE, 3, ps*2, ps*2)).to(device).float()
        in_img = in_img.view((BATCH_SIZE, 4, ps, ps)).to(device).float()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Get model outputs
        out = model(in_img)
        
        deteched_unary_potentials = out[0].detach().cpu().numpy()
        detached_rightNeighbor = out[1].detach().cpu().numpy() 
        detached_downNeighbor = out[2].detach().cpu().numpy() 
        detached_gt_imag = gt_img.detach().cpu().numpy()
        
#         pdb.set_trace()
        
        output, final_assignment_y = loss_augmented_map_inference(deteched_unary_potentials,
                                                                  detached_rightNeighbor,
                                                                  detached_downNeighbor,
                                                                  real_stuff=detached_gt_imag, is_real=True)
        output = torch.tensor(output)
        # _, output = torch.max(output, axis=2)
        # pixel_pots = torch.tensor(pixel_pots)
        # edge_pots = torch.tensor(edge_pots)
        labels = gt_img.float()
        final_assignment_y = torch.tensor(final_assignment_y).to(device).float()
        pixel_pots_b4 = out[0].to(device).float()
        edge_pots_b4_right = out[1].to(device).float()
        edge_pots_b4_down = out[2].to(device).float()
        output = output.to(device).float()
        loss = compute_struct_svm_loss(output, labels, pixel_pots_b4, edge_pots_b4_right, edge_pots_b4_down, final_assignment_y)
#         pdb.set_trace()
#         loss = loss.float()
        running_loss += loss.item()
        print(running_loss)
        
        loss.backward()
        optimizer.step()
        save_current_model_svm(model, epoch, output, gt_img[0], train_ids[0].item(), ratios[0].item())
        
#         if count % save_freq == 0 and count == 0:
#             save_current_model(model, epoch, out_img[0], gt_img[0], train_ids[0].item(), ratios[0].item())
#             count += 1

    return running_loss
            

