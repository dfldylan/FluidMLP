import numpy as np

target_vel = False

learning_rate = 0.001
batch = 16
dt = 0.004
voxel_size = 0.4
voxel_bias = np.transpose(np.where(np.ones([3, 3, 3]))) - 1  # (27, 3)

sample_folder = r'D:\dufeilong\data\sample\scene1'
save_folder = r'./save'
log_folder = r'./tensorboard'
if target_vel:
    save_folder += r'_vel'
    log_folder += r'_vel'

# readme
# vel or acc
# config.py -- save_folder, log_folder,
# pred.py -- write_csv
# files.py -- return build_data
# model.py -- return pred