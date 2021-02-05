from model import *
from dataset import *
import config as cfg
from hash import hash
import shutil

dt = 0.004
pred_fps = 200
start_csv = r'D:\dufeilong\data\sample\scene1\1\all_particles_300.csv'

if __name__ == '__main__':
    pool = Pool()

    gpus = tf.config.list_physical_devices(device_type='GPU')
    # tf.config.set_visible_devices(devices=gpus[1:2], device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    # load the dataset
    dataset = DataSet()

    # initial and import(optional) the model
    model = SlowFluidNet()
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(cfg.save_folder) is not None:
        latest_folder = tf.train.latest_checkpoint(cfg.save_folder)
        checkpoint.restore(latest_folder)
        global_step = int(os.path.split(latest_folder)[1].split(r'-')[0])
    else:
        print('no checkpoing found!')
        exit(-1)


    # prepare
    fps = int(start_csv.split(r'/')[-1].split(r'.')[0].split(r'_')[-1])
    id = hash(start_csv)
    pred_folder = r'./pred/' + id
    os.makedirs(pred_folder, exist_ok=True)
    shutil.copy(start_csv, pred_folder)
    df_log = pd.read_csv(r'./pred/log.csv')
    df_log = df_log.append({'csv_path':start_csv, 'id':id}, ignore_index=True)
    df_log.drop_duplicates().to_csv(r'./pred/log.csv', index=False)

    for i in range(pred_fps):
        csv_path = os.path.join(pred_folder, 'all_particles_' + str(fps) + '.csv')
        mask, center_particle, index, current_data = dataset.get_fps_full_particles(csv_path, pool)
        _inputs = mask, center_particle[:, :6], current_data[:, :7]
        _outputs = center_particle[:, 7:10]
        output = model(_inputs)

        # if acceleration
        # acc = output
        # vel = acc * dt + center_particle[:, 3:6]
        # if velocity
        vel = output

        pos = vel * dt + center_particle[:, :3]
        df = pd.read_csv(start_csv)
        df = df[df.isFluidSolid == 1]
        data_fluid = np.concatenate((pos, vel), axis=1)  # [-1, 6]
        df0 = pd.DataFrame(data_fluid, columns=df.columns[:6])
        df0[df.columns[6:18]] = 0
        df = df.append(df0)
        fps += 1
        df.to_csv(os.path.join(pred_folder, 'all_particles_' + str(fps) + '.csv'), index=False)

    print('ok!')
