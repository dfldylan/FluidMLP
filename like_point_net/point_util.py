import numpy as np
import tensorflow as tf


# Farthest Point Sampling
class FPS:
    def __init__(self, points):  # [-1, 3]
        self.points = points
        self.distance_to_set_each_particles = np.inf * tf.ones(shape=[points.shape[0]], dtype=float)

    def update_distance_to_set(self, new_particle_to_set_index):
        distance = tf.linalg.norm(self.points - self.points[new_particle_to_set_index], ord=2, axis=1)  # [-1,]
        self.distance_to_set_each_particles = tf.reduce_min([distance, self.distance_to_set_each_particles], axis=0)

    def get_farthest_particle_index(self):
        return tf.argmax(self.distance_to_set_each_particles)

    # def get_min_distance(self, a, b):
    #     distance = []
    #     for i in range(a.shape[0]):
    #         dis = np.sum(np.square(a[i] - b), axis=-1)
    #         distance.append(dis)
    #     distance = np.stack(distance, axis=-1)
    #     distance = np.min(distance, axis=-1)
    #     return np.argmax(distance), np.max(distance)

    # @staticmethod
    # def get_model_corners(model):
    #     min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
    #     min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
    #     min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
    #     corners_3d = np.array([
    #         [min_x, min_y, min_z],
    #         [min_x, min_y, max_z],
    #         [min_x, max_y, min_z],
    #         [min_x, max_y, max_z],
    #         [max_x, min_y, min_z],
    #         [max_x, min_y, max_z],
    #         [max_x, max_y, min_z],
    #         [max_x, max_y, max_z],
    #     ])
    #     return corners_3d

    def compute_fps(self, K):
        # 计算中心点位
        # corner_3d = self.get_model_corners(self.points)
        center_3d = tf.reduce_mean(self.points, axis=0)  # [3,]
        # A = np.array([center_3d])
        # B = np.array(self.points)
        t = []
        # 寻找Ｋ个节点
        for i in range(K):
            if i == 0:
                self.distance_to_set_each_particles = tf.linalg.norm(self.points - center_3d, ord=2, axis=1)
            else:
                self.update_distance_to_set(select_id)
            select_id = self.get_farthest_particle_index()
            # A = np.append(A, np.array([B[max_id]]), 0)
            # B = np.delete(B, max_id, 0)
            t.append(select_id)
        A = tf.Variable(tf.gather(self.points, t))
        self.update_distance_to_set(select_id)
        max_distance = tf.reduce_max(self.distance_to_set_each_particles)
        return A, t, max_distance


def get_center_particle_index(pos):  # [-1, 3]
    center_pos = tf.reduce_mean(pos, axis=0)
    indices = tf.argmin(tf.linalg.norm(pos - center_pos, ord=2, axis=1))
    return indices


def opt_group(indices, pos, radius):
    group_indices = []
    for index in indices:
        center_pos = pos[index]
        distance = tf.linalg.norm(pos - center_pos, ord=2, axis=1)  # [N]
        single_indices = tf.where(distance < radius)[:, 0]
        group_indices.append(single_indices)
    return group_indices  # [N] -> [M]
