import numpy as np
import config as cfg
from files import *
from multiprocessing.dummy import Pool


# import random

class DataSet(object):
    def __init__(self):
        sample_folder = cfg.sample_folder
        self.files = find_files(sample_folder)

    def next_particles(self, pool=None, batch=cfg.batch):
        self.current_file = random.choice(self.files)
        # print(self.current_file)
        self.current_data = build_data(get_data_from_file(self.current_file), voxel_size=cfg.voxel_size,
                                       target_vel=cfg.target_vel, dt=cfg.dt)
        # find fluid indices
        fluid_mask = self.current_data[:, 6] == 0
        self.fluid_indices = np.where(fluid_mask)[0]
        fluid_num = self.fluid_indices.shape[0]

        select_num = min(batch, fluid_num)
        ret = pool.map(self._find_neighbor, range(select_num))  # [(m1, cp1, i1), (m2, cp2, i2)...]
        return np.array([item[0] for item in ret]), np.array([item[1] for item in ret]), np.array(
            [item[2] for item in ret]), self.current_data

    def _find_neighbor(self, i):
        # find neighbor particles for one
        index = random.choice(self.fluid_indices)
        center_particle = self.current_data[index]
        center_voxel = center_particle[-3:]
        voxel_neigh = cfg.voxel_bias + center_voxel  # [27,3]
        voxel_all = self.current_data[:, -3:]  # [-1, 3]
        mask = np.any(np.all(voxel_neigh == np.expand_dims(voxel_all, axis=1), axis=2), axis=1)  # [-1]
        mask[index] = False
        return mask, center_particle, index

    def _find_neighbor_all(self, index):
        # find neighbor particles for one
        center_particle = self.current_data[index]
        center_voxel = center_particle[-3:]
        voxel_neigh = cfg.voxel_bias + center_voxel  # [27,3]
        voxel_all = self.current_data[:, -3:]  # [-1, 3]
        mask = np.any(np.all(voxel_neigh == np.expand_dims(voxel_all, axis=1), axis=2), axis=1)  # [-1]
        mask[index] = False
        return mask, center_particle, index

    # build every particles for specific csv_path
    def get_fps_full_particles(self, file_path, pool=None):
        self.current_file = file_path
        self.current_data = build_data(get_data_from_file(self.current_file), voxel_size=cfg.voxel_size,
                                       target_vel=cfg.target_vel, dt=cfg.dt)
        # find fluid indices
        fluid_mask = self.current_data[:, 6] == 0
        self.fluid_indices = np.where(fluid_mask)[0]
        # fluid_num = self.fluid_indices.shape[0]

        ret = pool.map(self._find_neighbor_all, self.fluid_indices)  # [(m1, cp1, i1), (m2, cp2, i2)...]
        return np.array([item[0] for item in ret]), np.array([item[1] for item in ret]), np.array(
            [item[2] for item in ret]), self.current_data


if __name__ == '__main__':
    dataset = DataSet()
    data = dataset.next_particles()
    # print(data)
