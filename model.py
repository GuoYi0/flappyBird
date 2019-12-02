import tensorflow as tf
import numpy as np


def init():
    return tf.initializers.truncated_normal(stddev=0.01)


def createNetwork(num_classes, is_training=True):
    inputs = tf.placeholder(tf.float32, [None, 128, 72, 4], name="images")  # 128x72x4
    f = tf.layers.conv2d(inputs, filters=32, kernel_size=7, strides=4, padding="SAME",
                         kernel_initializer=init(), name="conv1")  # 32x18x32
    f = tf.layers.batch_normalization(f, training=is_training, name="bn1")
    f = tf.nn.relu(f)
    f = tf.layers.conv2d(f, filters=64, kernel_size=3, strides=2, padding="SAME",
                         kernel_initializer=init(), name="conv2")  # 16x9x64
    f = tf.layers.batch_normalization(f, training=is_training, name="bn2")
    f = tf.nn.relu(f)
    f = tf.layers.conv2d(f, filters=128, kernel_size=3, strides=2, padding="SAME",
                         kernel_initializer=tf.initializers.variance_scaling(), name="conv3")  # 8x5x128
    f = tf.layers.batch_normalization(f, training=is_training, name="bn3")
    f = tf.nn.relu(f)
    f = tf.layers.conv2d(f, filters=256, kernel_size=3, strides=2, padding="SAME",
                         kernel_initializer=tf.initializers.variance_scaling(), name="conv4")  # 4x3x256
    f = tf.layers.batch_normalization(f, training=is_training, name="bn4")
    f = tf.nn.relu(f)
    f = tf.reshape(f, (tf.shape(f)[0], 3072), name="reshape2fc")
    f = tf.layers.dense(f, units=512, activation=tf.nn.relu,
                        kernel_initializer=tf.initializers.variance_scaling(), name="fc1")
    f = tf.layers.dense(f, units=num_classes, name="fc2",
                        kernel_initializer=tf.initializers.variance_scaling(), use_bias=False)

    # 这样人为初始化一个权重，是因为鸟很容易撞着天空了。我们给10一个较大的先验。
    final_bias = tf.get_variable(name="final_bias", dtype=tf.float32,
                                 initializer=np.array([[0.05, -0.05]], dtype=np.float32))
    outputs = f + final_bias
    outputs = tf.tanh(outputs/2, name="final_tanh")  # 约束在 (-1, 1)之间。这里是重大改动!!
    return inputs, outputs
