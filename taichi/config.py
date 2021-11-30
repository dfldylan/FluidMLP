import numpy as np
import os

LEARNING_RATE = 0.001
SAVE_FOLDER = r'./save'
LOG_FOLDER = r'./tensorboard'
SAMPLE_FOLDER = r'/root/datasets/new'
BATCH_SIZE = 64 * os.cpu_count()  # todo batch
VOXEL_SIZE = 0.4
TAEGET_VEL = False
DT = 1/30
RANDOM_ROTATE = False
VOXEL_BIAS = np.transpose(np.where(np.ones([3, 3, 3]))) - 1  # (27, 3)

if TAEGET_VEL:
    SAVE_FOLDER += r'_vel'
    LOG_FOLDER += r'_vel'

os.makedirs(SAVE_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# readme
# vel or acc
# config.py -- save_folder, log_folder,
# pred.py -- write_csv
# files.py -- return build_data
# model.py -- return pred
