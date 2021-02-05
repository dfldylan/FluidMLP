import numpy as np

learning_rate = 0.0001
batch = 16

voxel_size = 0.4
voxel_bias = np.transpose(np.where(np.ones([3, 3, 3]))) - 1

sample_folder = r'D:\dufeilong\data\sample\scene1'
save_folder = r'./save_vel'
log_folder = r'./tensorboard_vel'


# readme
# vel or acc
# config.py -- save_folder, log_folder,
# pred.py -- write_csv
# files.py -- return build_data
# model.py -- return pred