from model import *
from dataset_taichi import *
import config as cfg
from multiprocessing import Pool

global_step = -1

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
    optimizer = tf.optimizers.Adam(learning_rate=cfg.LEARNING_RATE)
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(cfg.SAVE_FOLDER) is not None:
        latest_folder = tf.train.latest_checkpoint(cfg.SAVE_FOLDER)
        checkpoint.restore(latest_folder)
        global_step = int(os.path.split(latest_folder)[1].split(r'-')[0])
    summary_writer = tf.summary.create_file_writer(cfg.LOG_FOLDER)

    while True:
        print('step ' + str(global_step + 1), end='...')
        mask, index, current_data = dataset.next_particles(pool)
        _inputs = mask, index, current_data[:, :7]
        _outputs = current_data[index, 7:10]
        current_loss = train(model, inputs=_inputs, outputs=_outputs, optimizer=optimizer)

        global_step += 1
        print(str(current_loss.numpy()), end='...')
        if global_step % 10 == 0:
            print('save', end='...')
            checkpoint.save(os.path.join(cfg.SAVE_FOLDER, str(global_step) + r'-model.ckpt'))
            with summary_writer.as_default():
                tf.summary.scalar("loss", current_loss, step=global_step)

        print('ok!')
