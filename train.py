from model import *
from dataset import *
import config as cfg

gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

# load the dataset
dataset = DataSet()

# initial and import(optional) the model
model = SlowFluidNet()
optimizer = tf.optimizers.Adam(learning_rate=cfg.learning_rate)
checkpoint = tf.train.Checkpoint(paras=model.variables)
if tf.train.latest_checkpoint(cfg.save_foler) is not None:
    checkpoint.restore(tf.train.latest_checkpoint(cfg.save_foler))

if __name__ == '__main__':
    step = 1
    while True:
        print('step ' + str(step) + '...', end='')
        mask, center_particle, index, current_data = dataset.next_particles()
        _inputs = [[mask[i], center_particle[i, :6]] for i in range(len(index))], current_data[:, :7]
        _outputs = center_particle[:, 7:10]
        train(model, inputs=_inputs, outputs=_outputs, optimizer=optimizer)
        print('ok!')
        while step % 10 == 0:
            checkpoint.save(os.path.join(cfg.save_foler, r'model.ckpt'))
