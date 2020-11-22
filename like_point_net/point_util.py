import numpy as np


# Farthest Point Sampling
class FPS:
    def __init__(self, points):  # [-1, 3]
        self.points = np.unique(points, axis=0)

    def get_min_distance(self, a, b):
        distance = []
        for i in range(a.shape[0]):
            dis = np.sum(np.square(a[i] - b), axis=-1)
            distance.append(dis)
        distance = np.stack(distance, axis=-1)
        distance = np.min(distance, axis=-1)
        return np.argmax(distance), np.max(distance)

    @staticmethod
    def get_model_corners(model):
        min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
        min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
        min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
        corners_3d = np.array([
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ])
        return corners_3d

    def compute_fps(self, K):
        # 计算中心点位
        corner_3d = self.get_model_corners(self.points)
        center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
        A = np.array([center_3d])
        B = np.array(self.points)
        t = []
        # 寻找Ｋ个节点
        for i in range(K):
            max_id, max_distance = self.get_min_distance(A, B)
            if i == 0:
                A = np.array([B[max_id]])
            else:
                A = np.append(A, np.array([B[max_id]]), 0)
            B = np.delete(B, max_id, 0)
            t.append(max_id)
        return A, t, max_distance


def get_center_particle_index(pos):  # [-1, 3]
    center_pos = np.average(pos, axis=0)
    indices = np.argmin(np.linalg.norm(pos - center_pos, ord=2, axis=1))
    return indices


def opt_group(indices, pos, radius):
    group_indices = []
    for index in indices:
        center_pos = pos[index]
        distance = np.linalg.norm(pos - center_pos, ord=2, axis=1)  # [N]
        single_indices = np.where(distance < radius)[0]
        group_indices.append(single_indices)
    return group_indices  # [N] -> [M]


def interpolate(known_pos_indices, known_feature, unknown_pos_indices, pos_array, out_channel):
    return current_data
