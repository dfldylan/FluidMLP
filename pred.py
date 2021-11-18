from model import *
from dataset import *
import config as cfg
from hash import hash
import shutil
import multiprocessing as mp

# import pdb

batch_size = 2 ** 24
pred_fps = 50
# start_csv = r'/root/datasets/mountain/all_particles_350.csv'
start_csv_list = [r'/root/datasets/large/all_particles_350.csv',
                  r'/root/datasets/large/all_particles_390.csv',
                  r'/root/datasets/normal/0/all_particles_300.csv',
                  r'/root/datasets/normal/171/all_particles_300.csv',
                  r'/root/datasets/normal/175/all_particles_300.csv']
output_folder_root = r'/root/datasets/pred/fluidsnet_lite'


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

    for start_csv in start_csv_list:
        # prepare
        fps = int(start_csv.split(r'/')[-1].split(r'.')[0].split(r'_')[-1])
        # fast mode start
        id = hash(start_csv)
        print(id, start_csv)
        pred_folder = r'./pred/' + id
        output_folder = os.path.join(output_folder_root, id)
        os.makedirs(pred_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
        shutil.copy(start_csv, pred_folder)
        if output_folder is not None:
            shutil.copy(start_csv, output_folder)
        df_log = pd.read_csv(r'./pred/log.csv')
        df_log = df_log.append({'csv_path': start_csv, 'id': id}, ignore_index=True)
        df_log.drop_duplicates().to_csv(r'./pred/log.csv', index=False)
        # fast mode end

        table_data = get_data_from_file(start_csv)
        for i in range(pred_fps):
            # calculate voxel index
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

            # concat csv data with solid particles
            df = pd.read_csv(start_csv)
            df = df[df.isFluidSolid == 1]
            data_fluid = np.concatenate((pos, vel), axis=1)  # [-1, 6]
            df0 = pd.DataFrame(data_fluid, columns=df.columns[:6])
            df0[df.columns[6:7]] = 0.004
            df0[df.columns[7:18]] = 0
            df = df.append(df0)
            fps += 1
            table_data = df.values

            print(str(fps) + ' ok!')
            # write csv -- fast mode start
            df.to_csv(os.path.join(pred_folder, 'all_particles_' + str(fps) + '.csv'), index=False)
            if output_folder is not None:
                shutil.copy(os.path.join(pred_folder, 'all_particles_' + str(fps) + '.csv'), output_folder)
            # fast mode end
        print('finished!')
