import tensorflow as tf
import numpy as np


def loss(pred, truth):
    return tf.reduce_mean(tf.square(pred - truth))


def train(model, inputs, outputs, optimizer):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    grads = t.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


class SlowFluidNet(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fluid_fcn0 = tf.keras.layers.Dense(18, input_shape=(None, 9), activation=tf.keras.activations.tanh)
        self.fluid_fcn1 = tf.keras.layers.Dense(9, input_shape=(None, 18), activation=tf.keras.activations.tanh)
        self.fluid_fcn2 = tf.keras.layers.Dense(6, input_shape=(None, 9), activation=tf.keras.activations.tanh)
        self.fluid_fcn3 = tf.keras.layers.Dense(3, input_shape=(None, 6))
        self.solid_fcn0 = tf.keras.layers.Dense(18, input_shape=(None, 6), activation=tf.keras.activations.tanh)
        self.solid_fcn1 = tf.keras.layers.Dense(9, input_shape=(None, 18), activation=tf.keras.activations.tanh)
        self.solid_fcn2 = tf.keras.layers.Dense(6, input_shape=(None, 9), activation=tf.keras.activations.tanh)
        self.solid_fcn3 = tf.keras.layers.Dense(3, input_shape=(None, 6))
        # self.variables = [self.fluid_fcn0, self.fluid_fcn1, self.fluid_fcn2, self.fluid_fcn3, self.solid_fcn0,
        #                   self.solid_fcn1, self.solid_fcn2, self.solid_fcn3]

    # def __call__(self, inputs):  # (-1, 6) (-1, 3) (6,)
    #     neigh_fluid, neigh_solid, center_property = inputs
    #     center_property = np.expand_dims(center_property, axis=0)  # (1, 6)
    #     neigh_fluid[:, :3] = neigh_fluid[:, :3] - center_property[:, :3]
    #     neigh_solid = neigh_solid - center_property[:, :3]
    #     neigh_fluid = np.concatenate((neigh_fluid, np.tile(center_property[:, 3:], (neigh_fluid.shape[0], 1))),
    #                                  axis=1)  # (-1, 9)
    #     neigh_solid = np.concatenate((neigh_solid, np.tile(center_property[:, 3:], (neigh_solid.shape[0], 1))),
    #                                  axis=1)  # (-1, 6)
    #
    #     x = self.fluid_fcn0(neigh_fluid)
    #     x = self.fluid_fcn1(x)
    #     x = self.fluid_fcn2(x)
    #     x = self.fluid_fcn3(x)
    #     y = self.solid_fcn0(neigh_solid)
    #     y = self.solid_fcn1(y)
    #     y = self.solid_fcn2(y)
    #     y = self.solid_fcn3(y)
    #     pred = tf.reduce_sum(tf.concat([x, y], axis=0), axis=0)  # (3,)
    #     return pred
    def call(self, inputs, training=None, mask=None):  # [(m1, cp1), (m2, cp2)...]
        current_particles, current_data = inputs
        self.data = current_data
        ret = tf.vectorized_map(self._each_particle, current_particles)  # [-1, 3]
        return ret

    def _each_particle(self, inputs):  # (m1, cp1)
        mask, center_particle = inputs
        center_particle = np.expand_dims(center_particle, axis=0)  # (1, 6)
        neighbor_particles = self.data[mask]
        neighbor_particles[:3] = neighbor_particles[:3] - center_particle[:3]
        fluid_part = neighbor_particles[neighbor_particles[:, 6] == 0, :6]
        solid_part = neighbor_particles[neighbor_particles[:, 6] == 1, :3]
        neigh_fluid = np.concatenate((fluid_part, np.tile(center_particle[:, 3:], (fluid_part.shape[0], 1))),
                                     axis=1)  # (-1, 9)
        neigh_solid = np.concatenate((solid_part, np.tile(center_particle[:, 3:], (solid_part.shape[0], 1))),
                                     axis=1)  # (-1, 6)
        x = self.fluid_fcn0(neigh_fluid)
        x = self.fluid_fcn1(x)
        x = self.fluid_fcn2(x)
        x = self.fluid_fcn3(x)
        y = self.solid_fcn0(neigh_solid)
        y = self.solid_fcn1(y)
        y = self.solid_fcn2(y)
        y = self.solid_fcn3(y)
        pred = tf.reduce_sum(tf.concat([x, y], axis=0), axis=0)  # (3,)
        return pred
