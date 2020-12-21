import numpy as np

learning_rate = 0.01
batch = 16

voxel_size = 0.4
voxel_bias = np.transpose(np.where(np.ones([3, 3, 3]))) - 1

sample_folder = r'D:\data\sample\scene1'
save_folder = r'./save'
log_folder = r'./tensorboard'
