import numpy as np
import os
import pandas as pd
import random


class RotateClass:
    def __init__(self, angel):
        self.sin_theta, self.cos_theta = np.sin(angel), np.cos(angel)

    def __call__(self, x, y):
        x = x * self.cos_theta - y * self.sin_theta
        y = x * self.sin_theta + y * self.cos_theta
        return x, y


def build_data(table_data, label_data=None, voxel_size=None, dt=None, target_vel=False, random_rotate=False):
    pos, vel, phase = table_data[:, :3], table_data[:, 3:6], table_data[:, 6:7]
    if not label_data:
        out = np.zeros_like(pos)
    else:
        l_pos, l_vel, l_phase = label_data[:, :3], label_data[:, 3:6], label_data[:, 6:7]
        if not np.all(np.equal(phase, l_phase)):
            print("change between two fps!")
            return None
        out = l_vel
        # accel instead of vel
        if not target_vel:
            out -= vel
            out /= dt

    if random_rotate:
        rotate = RotateClass(np.random.random() * 2 * np.pi)
        pos[:, 0], pos[:, 2] = rotate(pos[:, 0], pos[:, 2])
        vel[:, 0], vel[:, 2] = rotate(vel[:, 0], vel[:, 2])
        out[:, 0], out[:, 2] = rotate(out[:, 0], out[:, 2])

    if voxel_size:
        voxel = np.floor(pos * (1 / voxel_size)).astype(int)
    else:
        voxel = np.zeros_like(pos)
    data = np.concatenate([pos, vel, phase, out, voxel], axis=1)
    return data


def get_data_from_file(file_path):
    df = pd.read_csv(file_path, dtype=float)
    if df.isna().any(axis=None):
        print('nan exist')
        df = df[df.notna().all(axis=1)]
    return df.values


# def get_fps_from_filename_taichi(filename):
#     return int(os.path.basename(filename).split('.')[0])

def find_files(root_path, range_up=None, range_down=None, scene_num=None):
    folders = []
    exist_folders = [item for item in os.listdir(root_path) if len(item.split(r'.')) < 2]
    if scene_num is not None:
        for each in scene_num:
            each = str(each)
            if each in exist_folders:
                folders.append(each)
            else:
                print(f"folder {each} doesn't exist!")
    else:
        folders = exist_folders

    files_path = []
    for item in folders:
        path = os.path.join(root_path, item)
        files = [file for file in os.listdir(path) if file.split(r'.')[-1] == 'csv']
        if range_up is None and range_down is None:
            files_path += (os.path.join(path, file) for file in files)
        else:
            for file in files:
                fps = int(file.split(r'.')[0].split(r'_')[-1])
                if range_up is not None and fps >= range_up:
                    continue
                if range_down is not None and fps < range_down:
                    continue
                files_path.append(os.path.join(path, file))
    return files_path


if __name__ == '__main__':
    ret = get_data_from_file(r'D:\data\sample\scene1\1\all_particles_1.csv')
    print(ret)
