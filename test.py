import pandas as pd
from main.files import *
from main import config as cfg
import numpy as  np
from multiprocessing.dummy import Pool


def get_particle_neighbor_all(index):
    voxel_index = voxel_indices[index]  # (3,)
    near_voxel_index = cfg.voxel_bias + voxel_index  # (27, 3)
    keys_list = list(map(str, near_voxel_index))
    near_particles_index = []
    list(map(near_particles_index.extend, [voxel_dict[item] for item in keys_list if item in voxel_dict.keys()]))
    return near_particles_index, len(near_particles_index)


if __name__ == '__main__':
    pool = Pool()
    start_csv = r'D:\dufeilong\data\sample\scene1\1\all_particles_300.csv'
    table_data = get_data_from_file(start_csv)
    data = build_data(table_data, voxel_size=cfg.voxel_size, dt=cfg.dt, target_vel=cfg.target_vel)
    # build voxel dict
    voxel_indices = data[:, -3:]
    data_keys = list(map(str, voxel_indices))
    ds = pd.Series(data_keys)
    voxel_dict = dict(ds.groupby(ds).groups)

    # for each particle, get its neighbor particles' index and each neighbor counter.
    fluid_indices = np.where(data[:, 6] == 0)[0]




    ret = pool.map(get_particle_neighbor_all, fluid_indices)
    print(np.reshape(ret, [-1, 2]))
