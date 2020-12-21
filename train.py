from model import *
from dataset import *
import config as cfg

global_step = 0

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
    optimizer = tf.optimizers.Adam(learning_rate=cfg.learning_rate)
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(cfg.save_folder) is not None:
        latest_folder = tf.train.latest_checkpoint(cfg.save_folder)
        checkpoint.restore(latest_folder)
        global_step = int(os.path.split(latest_folder)[1].split(r'-')[0])
    summary_writer = tf.summary.create_file_writer(cfg.log_folder)

    while True:
        print('step ' + str(global_step), end='...')
        mask, center_particle, index, current_data = dataset.next_particles(pool)
        _inputs = mask, center_particle[:, :6], current_data[:, :7]
        _outputs = center_particle[:, 7:10]
        current_loss = train(model, inputs=_inputs, outputs=_outputs, optimizer=optimizer)
        global_step += 1
        if global_step % 10 == 0:
            print(str(current_loss.numpy()), end='...')
            checkpoint.save(os.path.join(cfg.save_folder, str(global_step) + r'-model.ckpt'))
            with summary_writer.as_default():
                tf.summary.scalar("loss", current_loss, step=global_step)
        print('ok!')
