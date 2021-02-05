import pandas as pd
import numpy as np
import os
import random


def build_data(table_data, voxel_size=None):
    pos, vel, s_f = table_data[:, :3], table_data[:, 3:6], table_data[:, 7:8]
    # out = table_data[:, 12:15] - table_data[:, 3:6]
    out = table_data[:, 12:15]

    if voxel_size:
        voxel = np.floor(pos * (1 / voxel_size)).astype(int)
    else:
        voxel = np.zeros_like(pos)
    data = np.concatenate([pos, vel, s_f, out, voxel], axis=1)
    return data


def get_data_from_file(file_path):
    df = pd.read_csv(file_path, dtype=float)
    df['isFluidSolid'] = df['isFluidSolid'].astype(int)
    return df.values


def find_files(root_path):
    folders_path = [os.path.join(root_path, item) for item in os.listdir(root_path) if len(item.split(r'.')) < 2]
    files_path = [os.path.join(item, file) for item in folders_path for file in os.listdir(item) if
                  file.split(r'.')[-1] == 'csv' and int(file.split(r'_')[-1].split(r'.')[0]) >= 300 and int(
                      file.split(r'_')[-1].split(r'.')[0]) < 500]
    # random.shuffle(files_path)
    return files_path


if __name__ == '__main__':
    ret = get_data_from_file(r'D:\data\sample\scene1\1\all_particles_1.csv')
    print(ret)
