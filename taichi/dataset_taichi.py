import os.path

import numpy as np
import config as cfg
from files_taichi import *
import os


# import random

class DataSet(object):
    def __init__(self):
        sample_folder = cfg.SAMPLE_FOLDER
        self.files = find_files(sample_folder)

    def next_particles(self, pool=None, batch=cfg.BATCH_SIZE):
        while True:
            select_file = random.choice(self.files)
            print(select_file)
            self.fps = int(os.path.basename(select_file).split('.')[0])
            label_file = os.path.join(os.path.dirname(select_file), str(self.fps+1)+'.csv')
            if os.path.exists(label_file):
                break


        self.current_data = build_data(get_data_from_file(select_file), get_data_from_file(label_file), voxel_size=cfg.VOXEL_SIZE,
                                       target_vel=cfg.TAEGET_VEL, dt=cfg.DT, random_rotate=cfg.RANDOM_ROTATE)
        # find fluid indices
        fluid_mask = self.current_data[:, 6] > 0
        self.fluid_indices = np.where(fluid_mask)[0]
        fluid_num = self.fluid_indices.shape[0]

        select_num = min(batch, fluid_num)
        ret = pool.map(self._find_neighbor_random, range(select_num))  # [(m1, cp1, i1), (m2, cp2, i2)...]
        mask, index = np.array([item[0] for item in ret]), np.array([item[1] for item in ret])
        return mask, index, self.current_data

    def _find_neighbor_random(self, i):
        index = random.choice(self.fluid_indices)
        return self._find_neighbor(index)

    def _find_neighbor(self, index):
        # find neighbor particles for one
        center_particle = self.current_data[index]
        center_voxel = center_particle[-3:]
        voxel_neigh = cfg.VOXEL_BIAS + center_voxel  # [27,3]
        voxel_all = self.current_data[:, -3:]  # [-1, 3]
        mask = np.any(np.all(voxel_neigh == np.expand_dims(voxel_all, axis=1), axis=2), axis=1)  # [-1]
        mask[index] = False
        return mask, index

    # build every particles for specific csv_path
    # def get_fps_full_particles(self, file_path, pool=None):
    #     self.current_file = file_path
    #     self.current_data = build_data(get_data_from_file(self.current_file), voxel_size=cfg.VOXEL_SIZE,
    #                                    target_vel=cfg.TAEGET_VEL, dt=cfg.DT)
    #     # find fluid indices
    #     fluid_mask = self.current_data[:, 6] != 1
    #     self.fluid_indices = np.where(fluid_mask)[0]
    #     # fluid_num = self.fluid_indices.shape[0]
    #
    #     ret = pool.map(self._find_neighbor, self.fluid_indices)  # [(m1, cp1, i1), (m2, cp2, i2)...]
    #     mask, index = np.array([item[0] for item in ret]), np.array([item[1] for item in ret])
    #     return mask, index, self.current_data


if __name__ == '__main__':
    dataset = DataSet()
    data = dataset.next_particles()
    # print(data)
