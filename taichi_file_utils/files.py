import numpy as np
import os
import pandas as pd


class RotateClass:
    def __init__(self):
        self.angel = np.random.random() * 2 * np.pi
        self.sin_theta, self.cos_theta = np.sin(self.angel), np.cos(self.angel)

    def __call__(self, x, y):
        x = x * self.cos_theta - y * self.sin_theta
        y = x * self.sin_theta + y * self.cos_theta
        return x, y


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
                fps = int(file.split(r'.')[0])
                if range_up is not None and fps >= range_up:
                    continue
                if range_down is not None and fps < range_down:
                    continue
                files_path.append(os.path.join(path, file))
    return files_path


def get_fps_from_filename(filename):
    return int(os.path.basename(filename).split('.')[0])


if __name__ == '__main__':
    ret = get_data_from_file(r'D:\data\sample\scene1\1\all_particles_1.csv')
    print(ret)
