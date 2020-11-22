import tensorflow as tf
from files import *
from like_point_net.point_util import *

sample_path = r'D:\data\sample\scene1'
channel_list = [128, 512, 1024, 2048]

# dataset
files_path = find_files(sample_path)

# initialize the model
optimizer = tf.optimizers.Adam(learning_rate=0.01)
conv1d_0 = tf.keras.layers.Conv1D(channel_list[0], [1], [1], activation=tf.nn.tanh)
conv1d_1 = tf.keras.layers.Conv1D(channel_list[1], [1], [1], activation=tf.nn.tanh)
conv1d_2 = tf.keras.layers.Conv1D(channel_list[2], [1], [1], activation=tf.nn.tanh)
conv1d_3 = tf.keras.layers.Conv1D(channel_list[3], [1], [1], activation=tf.nn.tanh)
conv1d = [conv1d_0, conv1d_1, conv1d_2, conv1d_3]

# train loop
step = 0
while True:
    # get a batch from dataset
    file_path = np.random.choice(files_path)
    data = get_data_from_file(file_path)
    feature_origin = data[:, :7]
    fluid_part = data[data[:, 6] == '0']
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
                group_particles = current_data[one_group_indices]  # todo test  # [M, F]
                group_particles[:, :3] = group_particles[:, :3] - center_pos
                # [M, F] -> [M, F'] -> [F']
                feature = conv1d[vision](tf.expand_dims(group_particles, axis=0))
                feature_list.append(tf.math.reduce_max(feature[0], axis=0))

            if vision != 3:
                current_data = tf.concat((current_data[indices, :3], tf.stack(feature_list)), axis=1)
            else:
                current_data = tf.stack(feature_list)
                break

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
                uk_pos = feature_list[vision - 2][unknown_pos_indices, :3]
                k_pos = feature_list[vision - 1][known_pos_indices, :3]
            if vision == 1:
                uk_pos = inputs[unknown_pos_indices, :3]
                k_pos = feature_list[vision - 1][known_pos_indices, :3]
            if vision == 0:
                uk_pos = inputs[unknown_pos_indices, :3]
                k_pos = inputs[known_pos_indices, :3]

            # todo interpolate
            # k_pos:[N, 3] k_feature:[N, F] uk_pos:[N', 3]  -> uk_feature:[N', F] -> [N', F']
            for each_uk_pos in uk_pos:
                # calculate distance with each k_pos
                # select particles within radius distance
                # weighted sum
                #

                pass




            # loss
        loss = tf.math.reduce_sum(tf.square(current_data - feature_origin))
    grads = t.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # print and save

    step += 1
