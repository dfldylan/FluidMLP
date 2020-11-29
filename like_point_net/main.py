from like_point_net.model import *
from files import *

learning_rate = 0.01
sample_path = r'D:\data\sample\scene1'
save_path = r'./like_point_net/save'
channel_list = [[64, 64, 128],
                [128, 256, 512],
                [512, 512, 1024],
                [1024, 1024, 2048]]
channel_re_list = [[2048, 1024],
                   [1024, 512],
                   [256, 128],
                   [64, 4]]

# dataset
files_path = find_files(sample_path)

# initialize the model
model = Model(learning_rate=learning_rate, channel_list=channel_list, channel_re_list=channel_re_list)

checkpoint = tf.train.Checkpoint(model=model)
if tf.train.latest_checkpoint(save_path) is not None:
    checkpoint.restore(tf.train.latest_checkpoint(save_path))

# train loop
step = 0
step_feature = []
while True:
    # get a batch from dataset
    file_path = np.random.choice(files_path)
    data = get_data_from_file(file_path)
    feature_origin = data[:, :7]
    # fluid_mask = data[:, 6] == 0

    # train model
    with tf.GradientTape() as t:
        model_inputs = feature_origin
        ret = model(model_inputs)
        current_data = ret
        # loss
        loss = tf.math.reduce_mean(tf.square(current_data - feature_origin[:, 3:7]))

    grads = t.gradient(loss, model.variables)
    model.optimizer.apply_gradients(zip(grads, model.variables))
    # print and save
    step += 1
    print(str(step) + ': ' + str(loss))
    while step % 10 == 0:
        checkpoint.save(os.path.join(save_path, r'model.ckpt'))
