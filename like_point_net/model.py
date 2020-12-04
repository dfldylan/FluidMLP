from abc import ABC

import tensorflow as tf
import itertools
from like_point_net.point_util import *


class Model(tf.keras.Model, ABC):
    def __init__(self, learning_rate, channel_list, channel_re_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.step = tf.Variable(0, trainable=False, dtype=tf.int32, name='step')
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.conv = []
        for vision in range(len(channel_list)):
            self.vision_conv = []
            for each_index in range(len(channel_list[vision])):
                self._conv = tf.keras.layers.Conv1D(channel_list[vision][each_index], [1], [1], activation=tf.nn.tanh)
                self.vision_conv.append(self._conv)
            self.conv.append(self.vision_conv)
        self.conv_re = []
        for vision in range(len(channel_re_list)):
            self.vision_conv = []
            for each_index in range(len(channel_re_list[vision])):
                self._conv = tf.keras.layers.Conv1D(channel_re_list[vision][each_index], [1], [1],
                                                    activation=tf.nn.tanh)
                self.vision_conv.append(self._conv)
            self.conv_re.append(self.vision_conv)

    def call(self, inputs, training=None, mask=None):
        feature_origin = inputs
        current_data = feature_origin
        particles_number = current_data.shape[0]
        vision_factor = np.power(particles_number, 1 / 4)
        sample_particle_indices_relative = [list(range(particles_number))]

        for vision in range(4):
            if vision == 3:
                indices = [get_center_particle_index(current_data[:3])]
                group_indices = np.reshape(range(current_data.shape[0]), [1, -1])

            else:
                # sample
                sample_particles_num = max(16, int(
                    np.ceil(current_data.shape[0] / vision_factor)))  # todo how many particles in every vision?
                _, indices, max_distance = FPS(current_data[:, :3]).compute_fps(sample_particles_num)  # [N]
                # group
                group_indices = opt_group(indices, current_data[:, :3], radius=2 * max_distance)  # [N] -> [M]

            sample_particle_indices_relative.append(indices)

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
                    feature = self.conv[vision][single_layer](feature)
                feature_list.append(tf.concat((center_pos, tf.math.reduce_max(feature[0], axis=0)), axis=0))

            current_data = tf.stack(feature_list)
            # self.step_feature.append(current_data)
        self.global_feature = current_data[:, 3:]  # [1, 2048]
        current_data = self.global_feature

        # calculate the absolute indices
        sample_particle_indices_absolute = sample_particle_indices_relative
        for vision in range(len(sample_particle_indices_absolute)):
            if vision == 0 or vision == 1:
                continue
            for p_index in range(len(sample_particle_indices_absolute[vision])):
                sample_particle_indices_absolute[vision][p_index] = sample_particle_indices_absolute[vision - 1][
                    sample_particle_indices_absolute[vision][p_index]]

        self.sample_particle_indices_absolute = sample_particle_indices_absolute

        # rebuild particles
        for vision in reversed(list(range(4))):
            known_pos_indices = sample_particle_indices_absolute[vision + 1]
            known_feature = current_data
            pos_array = inputs[:, :3]
            unknown_pos_indices = sample_particle_indices_absolute[vision]

            uk_pos = tf.gather(pos_array, unknown_pos_indices)
            k_pos = tf.gather(pos_array, known_pos_indices)

            # k_pos:[N, 3] k_feature:[N, F] uk_pos:[N', 3]  -> uk_feature:[N', F] -> [N', F']
            uk_feature_interpolate = []
            for each_uk_pos in uk_pos:  # [N']
                # calculate distance with each k_pos
                distance = tf.linalg.norm(k_pos - each_uk_pos, 2, axis=1)  # [N]
                if vision == 3:
                    weighted_sum = known_feature[0]
                else:
                    # select particles with knn(3)
                    select_indices = np.argpartition(distance, [0, 1, 2])[0:3]  # [3]
                    # weighted sum
                    weighted_sum = 0
                    factor = tf.Variable([1/tf.reduce_max([distance[i], 1e-10]) for i in select_indices])
                    factor_sum = tf.reduce_sum(factor)
                    factor = factor / factor_sum
                    for i in range(3):
                        weighted_sum = weighted_sum + known_feature[select_indices[i]] * factor[i]
                uk_feature_interpolate.append(weighted_sum)
            uk_feature = tf.expand_dims(tf.concat((uk_pos, tf.stack(uk_feature_interpolate)), axis=1), axis=0)   # [1, N', F]
            for single_layer in range(2):
                uk_feature = self.conv_re[3 - vision][single_layer](uk_feature)
            current_data = uk_feature[0]  # [N', F']
        return current_data
