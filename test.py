import tensorflow as tf
import numpy as np

ret = tf.concat([np.ones([5, 3]), np.ones([4])], axis=1)

print(ret)
