import pandas as pd
import numpy as np
import os
import random


def build_data(pos, vel, s_f, out, voxel_size):
    if voxel_size:
        voxel = np.floor(pos * (1 / voxel_size)).astype(int)
    else:
        voxel = np.zeros_like(pos)
    data = np.concatenate([pos, vel, s_f, out, voxel], axis=1)
    return data


def get_data_from_file(file_path, voxel_size=None):
    df = pd.read_csv(file_path, dtype=float)
    df['isFluidSolid'] = df['isFluidSolid'].astype(int)
    return build_data(df.iloc[:, :3].values, df.iloc[:, 3:6].values, df.iloc[:, 7:8].values,
                      df.iloc[:, 12:15].values - df.iloc[:, 3:6].values, voxel_size)


def find_files(root_path):
    folders_path = [os.path.join(root_path, item) for item in os.listdir(root_path) if len(item.split(r'.')) < 2]
    files_path = [os.path.join(item, file) for item in folders_path for file in os.listdir(item) if
                  file.split(r'.')[-1] == 'csv']
    # random.shuffle(files_path)
    return files_path


if __name__ == '__main__':
    ret = get_data_from_file(r'D:\data\sample\scene1\1\all_particles_1.csv')
    print(ret)
