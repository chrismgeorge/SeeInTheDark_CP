input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
result_dir = './result_Sony/'
model_dir = './saved_model/'
ps = 512 # patch size for training
save_freq = 100
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')