import tensorflow as tf
from files import *
from like_point_net.point_util import *
import itertools

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
optimizer = tf.optimizers.Adam(learning_rate=0.01)
conv = []
for vision in range(len(channel_list)):
    vision_conv = []
    for each_index in range(len(channel_list[vision])):
        _conv = tf.keras.layers.Conv1D(channel_list[vision][each_index], [1], [1], activation=tf.nn.tanh,
                                       dtype=tf.double)
        vision_conv.append(_conv)
    conv.append(vision_conv)
conv_re = []
for vision in range(len(channel_re_list)):
    vision_conv = []
    for each_index in range(len(channel_re_list[vision])):
        _conv = tf.keras.layers.Conv1D(channel_re_list[vision][each_index], [1], [1], activation=tf.nn.tanh,
                                       dtype=tf.double)
        vision_conv.append(_conv)
    conv_re.append(vision_conv)
tranable_paras = list(itertools.chain([conv, conv_re]))

checkpoint = tf.train.Checkpoint(paras=tranable_paras)
if tf.train.latest_checkpoint(save_path) is not None:
    checkpoint.restore(tf.train.latest_checkpoint(save_path))

# train loop
step = 0
step_feature = []
while True:
    # get a batch from dataset
    file_path = np.random.choice(files_path)
    data = get_data_from_file(file_path)[:100]
    feature_origin = data[:, :7]
    fluid_part = data[data[:, 6] == 0]
    inputs = fluid_part[:, :6]
    outputs = fluid_part[:, 7:10]

    # train model
    with tf.GradientTape() as t:

        current_data = feature_origin
        sample_particle_indices_list = []
        for vision in range(4):
            if vision == 3:
                indices = [get_center_particle_index(current_data[:3])]
                group_indices = np.reshape(range(current_data.shape[0]), [1, -1])

            else:
                # sample
                _, indices, max_distance = FPS(current_data[:, :3]).compute_fps(
                    int(np.ceil(current_data.shape[0] / 8)))  # [N]
                # group
                group_indices = opt_group(indices, current_data[:, :3], radius=max_distance)  # [N] -> [M]
            sample_particle_indices_list.append(indices)

            # transform and network
            feature_list = []
            for index, one_group_indices in zip(indices, group_indices):
                # select particles
                center_pos = current_data[index, :3]
                group_particles = tf.gather(current_data, one_group_indices)
                _group_particles = group_particles[:, :3] - center_pos
                group_particles = tf.concat((_group_particles, group_particles[:, 3:]), axis=1)
                # [M, F] -> [M, F'] -> [F']
                feature = tf.expand_dims(group_particles, axis=0)
                for single_layer in range(3):
                    feature = conv[vision][single_layer](feature)
                feature_list.append(tf.math.reduce_max(feature[0], axis=0))

            if vision != 3:
                current_data = tf.concat((tf.gather(current_data, indices)[:, :3], tf.stack(feature_list)), axis=1)
            else:
                current_data = tf.stack(feature_list)
                break
            step_feature.append(current_data)
        global_feature = current_data  # [1, F3]
        current_data = global_feature

        # rebuild particles
        for vision in reversed(list(range(4))):
            known_pos_indices = sample_particle_indices_list[vision]
            known_feature = current_data
            pos_array = inputs[:, :3]
            if vision == 0:
                unknown_pos_indices = list(range(inputs.shape[0])),
                out_channel = 7
            else:
                unknown_pos_indices = sample_particle_indices_list[vision - 1]
                out_channel = channel_list[vision - 1]

            if vision == 2 or vision == 3:
                uk_pos = tf.gather(step_feature[vision - 2], unknown_pos_indices)[:, :3]
                k_pos = tf.gather(step_feature[vision - 1], known_pos_indices)[:, :3]
            if vision == 1:
                uk_pos = inputs[unknown_pos_indices, :3]
                k_pos = tf.gather(step_feature[vision - 1], known_pos_indices)[:, :3]
            if vision == 0:
                uk_pos = inputs[unknown_pos_indices, :3]
                k_pos = inputs[known_pos_indices, :3]

            # k_pos:[N, 3] k_feature:[N, F] uk_pos:[N', 3]  -> uk_feature:[N', F] -> [N', F']
            uk_feature_interpolate = []
            for each_uk_pos in uk_pos:  # [N']
                # calculate distance with each k_pos
                distance = tf.linalg.norm(k_pos - each_uk_pos, 2, axis=1)  # [N]
                # select particles with knn(3)
                select_indices = np.argpartition(distance, [1, 2, 3])[1:4]  # [3]
                # weighted sum
                weighted_sum = 0
                for i in range(3):
                    weighted_sum = weighted_sum + known_feature[i] / distance[i]
                uk_feature_interpolate.append(weighted_sum)
            uk_feature = tf.expand_dims(tf.stack(uk_feature_interpolate), axis=0)  # [N', F]
            for single_layer in range(2):
                uk_feature = conv_re[vision][single_layer](uk_feature)
            current_data = uk_feature
        current_data = tf.concat((pos_array, current_data), axis=1)  # [N', F']

        # loss
        loss = tf.math.reduce_sum(tf.square(current_data - feature_origin))
    grads = t.gradient(loss, tranable_paras)
    optimizer.apply_gradients(zip(grads, tranable_paras))
    # print and save
    step += 1
    print(str(step) + ': ' + str(loss))
    while step % 10 == 0:
        checkpoint.save(os.path.join(save_path, r'model.ckpt'))
