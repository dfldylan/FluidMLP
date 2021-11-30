import numpy as np
import pandas as pd

from main.model import *
from main.dataset import *
from main import config as cfg
from utils.hash import hash
import shutil
import multiprocessing as mp

batch_size = 2 ** 24
pred_fps = 50
column_name = ['position_x','position_y','position_z','velocity_x','velocity_y','velocity_z',
               'timestep','isFluidSolid','source','out_position_x','out_position_y','out_position_z',
               'out_velocity_x','out_velocity_y','out_velocity_z','acceleration_x','acceleration_y','acceleration_z']

def get_particle_neighbor_all(inputs):
    index, voxel_indices, voxel_dict = inputs
    voxel_index = voxel_indices[index]  # (3,)
    near_voxel_index = cfg.voxel_bias + voxel_index  # (27, 3)
    keys_list = voxel_indices_to_str(near_voxel_index)
    neighb_particles_index = []
    list(map(neighb_particles_index.extend, [voxel_dict[item] for item in keys_list if item in voxel_dict.keys()]))
    return neighb_particles_index, len(neighb_particles_index)


def build_particle_data(inputs):
    fluid_particles_neighbor_indices, fluid_indices, data = inputs
    neighbor_data = data[fluid_particles_neighbor_indices, :7]  # (-1, 7)
    center_data = data[fluid_indices]  # (13,)
    neighbor_data[:, :3] -= center_data[:3]
    fluid_part = neighbor_data[neighbor_data[:, 6] == 0, :6]
    solid_part = neighbor_data[neighbor_data[:, 6] == 1, :3]
    fluid_part = np.concatenate((fluid_part, np.broadcast_to(center_data[3:6], (fluid_part.shape[0], 3))), axis=1)
    solid_part = np.concatenate((solid_part, np.broadcast_to(center_data[3:6], (solid_part.shape[0], 3))), axis=1)

    return fluid_part, solid_part, fluid_part.shape[0], solid_part.shape[0]


def add_sum(inputs):
    fluid, solid = inputs
    data = np.vstack([fluid, solid])
    # bottom = batch_fluid_counter[index]
    # up = batch_fluid_counter[index+1]
    return np.sum(data, axis=0)


def voxel_indices_to_str(voxel_indices):
    voxel_indices_c = np.char.array(voxel_indices.astype(np.int))
    return (voxel_indices_c[:, 0] + b'_' + voxel_indices_c[:, 1] + b'_' + voxel_indices_c[:, 2]).astype(np.str)

fluids_part = np.zeros([0, 6])


def generate_fluid():
    global fluids_part
    new_fluids_pos = 0.1*(np.transpose(np.where(np.ones([16, 16, 16]))) + np.array([-8, 5, 0]))
    new_fluids = np.hstack([new_fluids_pos, np.zeros([np.shape(new_fluids_pos)[0], 3])])
    fluids_part = np.vstack([fluids_part, new_fluids])

def get_solid_data(index):
    df = pd.read_csv('./glove_csv/'+str(int(index/5)+30)+'.csv', index_col=0)
    d = df.pop('1')
    df.insert(df.shape[1],"1",d)
    return 0.1*df.values - np.array([0, -10, 0])

def get_fluid_data():
    return fluids_part

def update_fluid_data(pos, vel):
    fluids_part[:, :3] = pos
    fluids_part[:, 3:6] = vel

def get_all_table_data(index):
    solid_data = get_solid_data(index)
    fluid_data = get_fluid_data()
    expand_fluid =  np.hstack([fluid_data,
                                   0.004*np.ones([np.shape(fluid_data)[0], 1]),
                                   np.zeros([np.shape(fluid_data)[0], 11])])
    expand_solid = np.hstack([solid_data, np.zeros([np.shape(solid_data)[0], 3]),
                                   0.004*np.ones([np.shape(solid_data)[0], 1]),
                                   np.ones([np.shape(solid_data)[0], 1]),
                                   np.zeros([np.shape(solid_data)[0], 10])])
    return np.vstack([expand_fluid, expand_solid])


if __name__ == '__main__':
    pool = mp.Pool()

    gpus = tf.config.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    # initial and import(optional) the model
    model = SlowFluidNet(trainable=False)
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(cfg.save_folder) is not None:
        latest_folder = tf.train.latest_checkpoint(cfg.save_folder)
        checkpoint.restore(latest_folder)
    else:
        print('no checkpoing found!')
        exit(-1)

    for _ in range(250):
        if _ % 250 == 0:
            generate_fluid()
        table_data = get_all_table_data(_)
        pd.DataFrame(table_data, columns=column_name).to_csv('./glove_pred/'+str(_)+'.csv',index=False)
        data = build_data(table_data, voxel_size=cfg.voxel_size, dt=cfg.dt, target_vel=cfg.target_vel, no_out=True)
        # build voxel dict
        voxel_indices = data[:, -3:]
        data_keys = voxel_indices_to_str(voxel_indices)
        ds = pd.Series(data_keys)
        voxel_dict = dict(ds.groupby(ds).groups)

        # for each fluid particle, get its neighbor particles' index and each neighbor counter.
        fluid_indices_mask = data[:, 6] == 0
        fluid_indices = np.where(fluid_indices_mask)[0]


        # pdb.set_trace()
        def iter_get_particle_neighbor_all(fluid_indices, voxel_indices, voxel_dict):
            for index in fluid_indices:
                yield index, voxel_indices, voxel_dict

        # print(fluid_indices, voxel_indices, voxel_dict)
        ret = pool.map(get_particle_neighbor_all,
                       iter_get_particle_neighbor_all(fluid_indices, voxel_indices, voxel_dict))
        # ret = [each.get() for each in pool_list]
        [fluid_particles_neighbor_indices, neighbor_count] = list(map(list, zip(*ret)))
        # pdb.set_trace()

        final_flag = False
        result_list = []
        current_index = 0
        while True:
            # calculate end_index
            _neighbor_count_cum = np.cumsum(neighbor_count[current_index:])
            batch_mask = _neighbor_count_cum > batch_size
            current_index_end = 0
            if not np.any(batch_mask):
                current_index_end = len(neighbor_count)
                final_flag = True
            elif np.all(batch_mask):
                print('batch size is so small that contain one particle calculation!')
                exit(-1)
            else:
                current_index_end = current_index + np.where(batch_mask)[0][0]


            # pdb.set_trace()
            def iter_build_particle_data(current_index, current_index_end, fluid_particles_neighbor_indices,
                                         fluid_indices, data):
                for index in range(current_index, current_index_end):
                    yield fluid_particles_neighbor_indices[index], fluid_indices[index], data


            ret = pool.map(build_particle_data, iter_build_particle_data(current_index, current_index_end,
                                                                         fluid_particles_neighbor_indices,
                                                                         fluid_indices, data))
            # ret = [each.get() for each in pool_list]
            [batch_fluid_data, batch_solid_data, batch_fluid_counter, batch_solid_counter] = list(
                map(list, zip(*ret)))
            # pdb.set_trace()

            ret_fluid, ret_solid = model.pred(fluid_part=np.vstack(batch_fluid_data),
                                              solid_part=np.vstack(batch_solid_data))
            ret_fluid, ret_solid = ret_fluid.numpy(), ret_solid.numpy()
            # split ret via batch_counter to ret_list  # [(-1,3), (-1,3) ...]
            batch_fluid_counter = np.cumsum(batch_fluid_counter)
            batch_solid_counter = np.cumsum(batch_solid_counter)

            ret_fluid = np.split(ret_fluid, batch_fluid_counter)
            ret_solid = np.split(ret_solid, batch_solid_counter)
            # add sum (-1, 3) and append to result_list
            ret = pool.map(add_sum, zip(ret_fluid, ret_solid))
            result_list.extend(ret[:-1])

            if final_flag:
                break
            # update index
            current_index = current_index_end

        result = np.vstack(result_list)
        # add gravity if pred is accel
        # calculate vel and pos
        fluid_part = data[fluid_indices_mask]
        if not cfg.target_vel:
            accel = result + np.array([0, -9.8, 0])
            vel = accel * cfg.dt + fluid_part[:, 3:6]
        else:
            vel = result
        pos = vel * cfg.dt + fluid_part[:, :3]
        update_fluid_data(pos, vel)

        # concat csv data with solid particles
        # table_data = get_all_table_data(_)
        # df = pd.read_csv(start_csv)
        # df = df[df.isFluidSolid == 1]
        # data_fluid = np.hstack([pos, vel])  # [-1, 6]
        # df0 = pd.DataFrame(data_fluid, columns=column_name[:6])
        # df0[column_name[6:7]] = 0.004
        # df0[column_name[7:18]] = 0
        # df = df.append(df0)
        # fps += 1
        # table_data = df.values
