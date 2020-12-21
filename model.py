import tensorflow as tf
import numpy as np


def loss(pred, truth):
    return tf.reduce_mean(tf.square(pred - truth))


def train(model, inputs, outputs, optimizer):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
        grads = t.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # model.global_step = model.global_step + 1
    return current_loss


class SlowFluidNet(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.global_step = tf.Variable(initial_value=0, dtype=tf.int64, name='global_step')
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
        mask, center_particle, current_data = inputs
        self.data = current_data
        ret = tf.map_fn(self._each_particle, (mask, center_particle), dtype=tf.float32)  # [-1, 3]
        return tf.stack(ret)

    def _each_particle(self, inputs):  # (m1, cp1)
        mask, center_particle = inputs
        center_particle = tf.expand_dims(center_particle, axis=0)  # (1, 6)
        neighbor_particles = self.data[mask]
        t = neighbor_particles[:, :3] - center_particle[:, :3]
        neighbor_particles = tf.concat((t, neighbor_particles[:, 3:]), axis=1)
        fluid_part = neighbor_particles[tf.equal(tf.cast(neighbor_particles[:, 6], tf.int32), tf.constant(0))][:, :6]
        solid_part = neighbor_particles[tf.equal(tf.cast(neighbor_particles[:, 6], tf.int32), tf.constant(1))][:, :3]
        neigh_fluid = tf.concat((fluid_part, tf.tile(center_particle[:, 3:], (tf.shape(fluid_part)[0], 1))),
                                axis=1)  # (-1, 9)
        neigh_solid = tf.concat((solid_part, tf.tile(center_particle[:, 3:], (tf.shape(solid_part)[0], 1))),
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
