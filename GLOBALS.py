import torch

# Directories
input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
result_dir = './result_Sony/'
model_dir = './saved_model/'

# Special testing
m_path = './saved_model/'
m_name = 'checkpoint_sony_e4000.pth'
test_result_dir = './test_result_Sony/'

# Globals
ps = 256 # patch size (training)
save_freq = 500
BATCH_SIZE = 1

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')