import tensorflow as tf
import numpy as np
import config as cfg


def loss(pred, truth):
    return tf.reduce_mean(tf.square(pred - truth))


def pred(model, inputs):
    return model(inputs)


def train(model, inputs, outputs, optimizer):
    with tf.GradientTape() as t:
        pred = model(inputs)
        current_loss = loss(pred, outputs)
        grads = t.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # model.global_step = model.global_step + 1
    return current_loss


class SlowFluidNet(tf.keras.Model):
    def __init__(self, trainable=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.global_step = tf.Variable(initial_value=0, dtype=tf.int64, name='global_step')
        self.fluid_fcn0 = tf.keras.layers.Dense(18, input_shape=(None, 9), activation=tf.keras.activations.tanh, trainable=trainable)
        self.fluid_fcn1 = tf.keras.layers.Dense(9, input_shape=(None, 18), activation=tf.keras.activations.tanh, trainable=trainable)
        self.fluid_fcn2 = tf.keras.layers.Dense(6, input_shape=(None, 9), activation=tf.keras.activations.tanh, trainable=trainable)
        self.fluid_fcn3 = tf.keras.layers.Dense(3, input_shape=(None, 6), trainable=trainable)
        self.solid_fcn0 = tf.keras.layers.Dense(18, input_shape=(None, 6), activation=tf.keras.activations.tanh, trainable=trainable)
        self.solid_fcn1 = tf.keras.layers.Dense(9, input_shape=(None, 18), activation=tf.keras.activations.tanh, trainable=trainable)
        self.solid_fcn2 = tf.keras.layers.Dense(6, input_shape=(None, 9), activation=tf.keras.activations.tanh, trainable=trainable)
        self.solid_fcn3 = tf.keras.layers.Dense(3, input_shape=(None, 6), trainable=trainable)

    def call(self, inputs, training=None, mask=None):  # [(m1, cp1), (m2, cp2)...]
        mask, index, current_data = inputs
        self.data = current_data
        ret = tf.map_fn(self._each_particle, (mask, index), dtype=tf.float32)  # [-1, 3]
        return ret

    def _each_particle(self, inputs):  # (m1, cp1)
        mask, index = inputs
        center_particle = tf.gather(self.data, [index])  # (1, 7)
        neighbor_particles = self.data[mask]
        t = neighbor_particles[:, :3] - center_particle[:, :3]
        neighbor_particles = tf.concat((t, neighbor_particles[:, 3:]), axis=1)  # (-1, 7)
        fluid_part = neighbor_particles[tf.greater(neighbor_particles[:, 6], 0)]  # (-1, 7)
        solid_part = neighbor_particles[tf.less(neighbor_particles[:, 6], 1)][:, :3]  # (-1, 3)
        neigh_fluid = tf.concat((fluid_part, tf.tile(center_particle[:, 3:], (tf.shape(fluid_part)[0], 1))),
                                axis=1)  # (-1, 11)
        neigh_solid = tf.concat((solid_part, tf.tile(center_particle[:, 3:], (tf.shape(solid_part)[0], 1))),
                                axis=1)  # (-1, 7)

        x = self.fluid_fcn0(neigh_fluid)
        x = self.fluid_fcn1(x)
        x = self.fluid_fcn2(x)
        x = self.fluid_fcn3(x)
        y = self.solid_fcn0(neigh_solid)
        y = self.solid_fcn1(y)
        y = self.solid_fcn2(y)
        y = self.solid_fcn3(y)
        pred = tf.reduce_sum(tf.concat([x, y], axis=0), axis=0)  # (3,)
        # if acc instead of vel
        if not cfg.TAEGET_VEL:
            pred = tf.constant([0, -9.8, 0], shape=[3, ]) + pred  # (3,)
        return pred

    @tf.function(experimental_relax_shapes=True)
    def pred(self, fluid_part, solid_part):
        with tf.device('/gpu:0'):
            y = self.solid_fcn0(solid_part)
            y = self.solid_fcn1(y)
            y = self.solid_fcn2(y)
            y = self.solid_fcn3(y)

        with tf.device('/gpu:1'):
            x = self.fluid_fcn0(fluid_part)
            x = self.fluid_fcn1(x)
            x = self.fluid_fcn2(x)
            x = self.fluid_fcn3(x)
        return x, y
