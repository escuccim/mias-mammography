import numpy as np
import os
import wget
from sklearn.cross_validation  import train_test_split
import tensorflow as tf
from training_utils import download_file, get_batches, read_and_decode_single_example, load_validation_data, download_data

download_data()
# ## Create Model

# config
epochs = 100                  
batch_size = 64

## Hyperparameters
# Small epsilon value for the BN transform
epsilon = 1e-8

# learning rate
epochs_per_decay = 5
starting_rate = 0.001
decay_factor = 0.85
staircase = True

# learning rate decay variables
steps_per_epoch = int(26772 / batch_size)
print("Steps per epoch:", steps_per_epoch)

# lambdas
lamC = 0.00000
lamF = 0.00100

# use dropout
dropout = True
fcdropout_rate = 0.5
convdropout_rate = 0.0
pooldropout_rate = 0.2

num_classes = 2

train_path_0 = os.path.join("data", "training_0.tfrecords")
train_path_1 = os.path.join("data", "training_1.tfrecords")
train_path_2 = os.path.join("data", "training_2.tfrecords")
train_path_3 = os.path.join("data", "training_3.tfrecords")
test_path = os.path.join("data", "test.tfrecords")
train_files = [train_path_0, train_path_1, train_path_2, train_path_3]

## Build the graph
graph = tf.Graph()

# whether to retrain model from scratch or use saved model
init = True
model_name = "model_s0.0.4.05"
# 0.0.3.01 - using inception input stem
# 0.0.3.02 - removed conv layers after 4 as data was being downsized too much
# 0.0.3.03 - added Inception Block A
# 0.0.3.05 - added reduce section from Inception
# 0.0.3.06 - added block b and another reduce
# 0.0.3.07 - changed last max pool to average pool
# 0.0.4.01 - adding block c from inception
# 0.0.4.02 - organizing namespaces so as to better view graph
# 0.0.4.03 - changed conv1 to stride 1 followed by max pool
# 0.0.4.04 - tried to figure out global average pooling, ended up replacing flatten with reduce_mean over axes 1 and 2
# 0.0.4.05 - lowered learning rate to see if it will help learn faster

with graph.as_default():
    training = tf.placeholder(dtype=tf.bool, name="is_training")
    is_testing = tf.placeholder(dtype=bool, shape=(), name="is_testing")

    # create global step for decaying learning rate
    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(starting_rate,
                                               global_step,
                                               steps_per_epoch * epochs_per_decay,
                                               decay_factor,
                                               staircase=staircase)

    with tf.name_scope('inputs') as scope:
        image, label = read_and_decode_single_example(train_files, label_type="label_normal", normalize=False)

        X_def, y_def = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=2000,
                                              min_after_dequeue=1000)

        # Placeholders
        X = tf.placeholder_with_default(X_def, shape=[None, 299, 299, 1])
        y = tf.placeholder_with_default(y_def, shape=[None])

        X = tf.cast(X, dtype=tf.float32)

    # Convolutional layer 1
    with tf.name_scope('conv1.0') as scope:
        conv1 = tf.layers.conv2d(
            X,  # Input data
            filters=32,  # 32 filters
            kernel_size=(3, 3),  # Kernel size: 5x5
            strides=(2, 2),  # Stride: 2
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=100),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv1.0'
        )

        conv1 = tf.layers.batch_normalization(
            conv1,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn1.0'
        )

        # apply relu
        conv1 = tf.nn.relu(conv1, name='relu1.0')

        conv1 = tf.layers.max_pooling2d(
            conv1,  # Input
            pool_size=(3, 3),  # Pool size: 3x3
            strides=(2, 2),  # Stride: 2
            padding='SAME',  # "same" padding
            name='pool1.0'
        )

    with tf.name_scope('conv1.1') as scope:
        conv11 = tf.layers.conv2d(
            conv1,  # Input data
            filters=32,  # 32 filters
            kernel_size=(3, 3),  # Kernel size: 5x5
            strides=(1, 1),  # Stride: 2
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=300),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv1.1'
        )

        conv11 = tf.layers.batch_normalization(
            conv11,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn1.1'
        )

        # apply relu
        conv11 = tf.nn.relu(conv11, name='relu1.1')

        # optional dropout
        if dropout:
            conv11 = tf.layers.dropout(conv11, rate=convdropout_rate, seed=400, training=training)

    with tf.name_scope('conv1.2') as scope:
        conv12 = tf.layers.conv2d(
            conv11,  # Input data
            filters=64,  # 32 filters
            kernel_size=(3, 3),  # Kernel size: 5x5
            strides=(1, 1),  # Stride: 2
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=500),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv1.2'
        )

        conv12 = tf.layers.batch_normalization(
            conv12,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn1.2'
        )

        # apply relu
        conv12 = tf.nn.relu(conv12, name='relu1.2')

        # optional dropout
        if dropout:
            conv12 = tf.layers.dropout(conv12, rate=convdropout_rate, seed=600, training=training)

    # Max pooling layer 1.1
    with tf.name_scope('pool1.1') as scope:
        pool11 = tf.layers.max_pooling2d(
            conv12,  # Input
            pool_size=(3, 3),  # Pool size: 3x3
            strides=(2, 2),  # Stride: 2
            padding='SAME',  # "same" padding
            name='pool1.1'
        )

        pool12 = tf.layers.conv2d(
            conv12,  # Input data
            filters=96,  # 32 filters
            kernel_size=(3, 3),  # Kernel size: 5x5
            strides=(2, 2),  # Stride: 2
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=800),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='pool1.2'
        )

        pool12 = tf.layers.batch_normalization(
            pool12,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_pool1.2'
        )

        # apply relu
        pool12 = tf.nn.relu(pool12, name='relu_pool_1.2')

    # concatenate 1
    with tf.name_scope("concat1") as scope:
        concat1 = tf.concat(
            [pool11, pool12],
            axis=3,
            name='concat1'
        )

    ## Input branch 2.1
    with tf.name_scope('conv2.1.1') as scope:
        conv211 = tf.layers.conv2d(
            concat1,  # Input data
            filters=64,  # 32 filters
            kernel_size=(1, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=801),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv2.1.1'
        )

        conv211 = tf.layers.batch_normalization(
            conv211,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn2.1.1'
        )

        # apply relu
        conv211 = tf.nn.relu(conv211, name='relu2.1.1')

    with tf.name_scope('conv2.1.2') as scope:
        conv212 = tf.layers.conv2d(
            conv211,  # Input data
            filters=96,  # 32 filters
            kernel_size=(3, 3),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=802),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv2.1.2'
        )

        conv212 = tf.layers.batch_normalization(
            conv212,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn2.1.2'
        )

        # apply relu
        conv212 = tf.nn.relu(conv212, name='relu2.1.2')

    ## Input branch 2.2
    with tf.name_scope('conv2.2.1') as scope:
        conv221 = tf.layers.conv2d(
            concat1,  # Input data
            filters=64,  # 32 filters
            kernel_size=(1, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=803),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv2.2.1'
        )

        conv221 = tf.layers.batch_normalization(
            conv221,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn2.2.1'
        )

        # apply relu
        conv221 = tf.nn.relu(conv221, name='relu2.2.1')

    with tf.name_scope('conv2.2.2') as scope:
        conv222 = tf.layers.conv2d(
            conv221,  # Input data
            filters=64,  # 32 filters
            kernel_size=(7, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=804),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv2.2.2'
        )

        conv222 = tf.layers.batch_normalization(
            conv222,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn2.2.2'
        )

        # apply relu
        conv222 = tf.nn.relu(conv221, name='relu2.2.2')

    with tf.name_scope('conv2.2.3') as scope:
        conv223 = tf.layers.conv2d(
            conv222,  # Input data
            filters=64,  # 32 filters
            kernel_size=(1, 7),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=805),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv2.2.3'
        )

        conv223 = tf.layers.batch_normalization(
            conv223,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn2.2.3'
        )

        # apply relu
        conv223 = tf.nn.relu(conv223, name='relu2.2.3')

    with tf.name_scope('conv2.2.4') as scope:
        conv224 = tf.layers.conv2d(
            conv223,  # Input data
            filters=96,  # 32 filters
            kernel_size=(1, 7),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=806),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv2.2.4'
        )

        conv224 = tf.layers.batch_normalization(
            conv224,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn2.2.4'
        )

        # apply relu
        conv224 = tf.nn.relu(conv224, name='relu2.2.4')

    # concatenate 2
    with tf.name_scope("concat2") as scope:
        concat2 = tf.concat(
            [conv212, conv224],
            axis=3,
            name='concat2'
        )

    # Max pooling layer 2
    with tf.name_scope('pool2.1') as scope:
        pool21 = tf.layers.max_pooling2d(
            concat2,  # Input
            pool_size=(2, 2),  # Pool size: 3x3
            strides=(2, 2),  # Stride: 2
            padding='SAME',  # "same" padding
            name='pool2.1'
        )

        pool22 = tf.layers.conv2d(
            concat2,  # Input data
            filters=192,  # 32 filters
            kernel_size=(3, 3),  # Kernel size: 5x5
            strides=(2, 2),  # Stride: 2
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=807),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='pool2.2'
        )

        pool22 = tf.layers.batch_normalization(
            pool22,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_pool2.2'
        )

        # apply relu
        pool22 = tf.nn.relu(pool22, name='relu_pool_2.2')

    # concatenate 3
    with tf.name_scope("concat3") as scope:
        concat3 = tf.concat(
            [pool21, pool22],
            axis=3,
            name='concat3'
        )

    ## Block A
    with tf.name_scope('block_a_branch_1') as scope:
        pool_a_1 = tf.layers.average_pooling2d(
            concat3,  # Input
            pool_size=(2, 2),  # Pool size: 3x3
            strides=(1, 1),  # Stride: 2
            padding='SAME',  # "same" padding
            name='pool_a_1_1'
        )

        conv_a_1 = tf.layers.conv2d(
            pool_a_1,  # Input data
            filters=96,  # 32 filters
            kernel_size=(1, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=878),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_a_1_1'
        )

        conv_a_1 = tf.layers.batch_normalization(
            conv_a_1,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_a_1_1'
        )

        # apply relu
        conv_a_1 = tf.nn.relu(conv_a_1, name='relu_a_1_1')

    with tf.name_scope('block_a_branch_2') as scope:
        conv_a_2 = tf.layers.conv2d(
            concat3,  # Input data
            filters=96,  # 32 filters
            kernel_size=(1, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=808),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_a_1_2'
        )

        conv_a_2 = tf.layers.batch_normalization(
            conv_a_2,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_a_2_1'
        )

        # apply relu
        conv_a_2 = tf.nn.relu(conv_a_2, name='relu_a_2_1')

    with tf.name_scope('block_a_branch_3') as scope:
        conv_a_3_1 = tf.layers.conv2d(
            concat3,  # Input data
            filters=64,  # 32 filters
            kernel_size=(1, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=809),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_a_3_1'
        )

        conv_a_3_1 = tf.layers.batch_normalization(
            conv_a_3_1,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_a_3_1'
        )

        # apply relu
        conv_a_3_1 = tf.nn.relu(conv_a_3_1, name='relu_a_3_1')

        conv_a_3_2 = tf.layers.conv2d(
            conv_a_3_1,  # Input data
            filters=96,  # 32 filters
            kernel_size=(3, 3),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=820),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_a_3_2'
        )

        conv_a_3_2 = tf.layers.batch_normalization(
            conv_a_3_2,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_a_3_2'
        )

        # apply relu
        conv_a_3_2 = tf.nn.relu(conv_a_3_2, name='relu_a_3_2')

    with tf.name_scope('block_a_branch_4') as scope:
        conv_a_4_1 = tf.layers.conv2d(
            concat3,  # Input data
            filters=64,  # 32 filters
            kernel_size=(1, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=821),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_a_4_1'
        )

        conv_a_4_1 = tf.layers.batch_normalization(
            conv_a_4_1,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_a_4_1'
        )

        # apply relu
        conv_a_4_1 = tf.nn.relu(conv_a_4_1, name='relu_a_4_1')

        conv_a_4_2 = tf.layers.conv2d(
            conv_a_4_1,  # Input data
            filters=96,  # 32 filters
            kernel_size=(3, 3),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=822),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_a_4_2'
        )

        conv_a_4_2 = tf.layers.batch_normalization(
            conv_a_4_2,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_a_4_2'
        )

        # apply relu
        conv_a_4_2 = tf.nn.relu(conv_a_4_2, name='relu_a_4_2')

        conv_a_4_3 = tf.layers.conv2d(
            conv_a_4_2,  # Input data
            filters=96,  # 32 filters
            kernel_size=(3, 3),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=823),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_a_4_3'
        )

        conv_a_4_3 = tf.layers.batch_normalization(
            conv_a_4_3,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_a_4_3'
        )

        # apply relu
        conv_a_4_3 = tf.nn.relu(conv_a_4_3, name='relu_a_4_3')

    with tf.name_scope('concat_a') as scope:
        concat4 = tf.concat(
            [conv_a_1, conv_a_2, conv_a_3_2, conv_a_4_3],
            axis=3,
            name='concat_a_1'
        )

    with tf.name_scope('reduce_a_1') as scope:
        ## normal max pool
        pool_a_1_1 = tf.layers.max_pooling2d(
            concat4,  # Input
            pool_size=(3, 3),  # Pool size: 3x3
            strides=(2, 2),  # Stride: 2
            padding='SAME',  # "same" padding
            name='poola_1_1'
        )

    with tf.name_scope('reduce_a_2') as scope:
        ## conv with stride 2
        pool_a_1_2 = tf.layers.conv2d(
            concat4,  # Input data
            filters=128,  # 32 filters
            kernel_size=(3, 3),  # Kernel size: 9x9
            strides=(2, 2),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=824),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='pool_a_1_2'
        )

        pool_a_1_2 = tf.layers.batch_normalization(
            pool_a_1_2,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_pool_a_1_2'
        )

        # apply relu
        pool_a_1_2 = tf.nn.relu(pool_a_1_2, name='relu_pool_a_1_2')

    with tf.name_scope('reduce_a_3') as scope:
        ## multiple convs
        pool_a_1_3 = tf.layers.conv2d(
            concat4,  # Input data
            filters=96,  # 32 filters
            kernel_size=(1, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=825),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='pool_a_1_3'
        )

        pool_a_1_3 = tf.layers.batch_normalization(
            pool_a_1_3,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_pool_a_1_3'
        )

        # apply relu
        pool_a_1_3 = tf.nn.relu(pool_a_1_3, name='relu_pool_a_1_3')

        pool_a_1_3 = tf.layers.conv2d(
            pool_a_1_3,  # Input data
            filters=128,  # 32 filters
            kernel_size=(3, 3),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=826),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='pool_a_1_3_2'
        )

        pool_a_1_3 = tf.layers.batch_normalization(
            pool_a_1_3,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_pool_a_1_3_2'
        )

        # apply relu
        pool_a_1_3 = tf.nn.relu(pool_a_1_3, name='relu_pool_a_1_3_2')

        pool_a_1_3 = tf.layers.conv2d(
            pool_a_1_3,  # Input data
            filters=192,  # 32 filters
            kernel_size=(3, 3),  # Kernel size: 9x9
            strides=(2, 2),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=827),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='pool_a_1_3_3'
        )

        pool_a_1_3 = tf.layers.batch_normalization(
            pool_a_1_3,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_pool_a_1_3_3'
        )

        # apply relu
        pool_a_1_3 = tf.nn.relu(pool_a_1_3, name='relu_pool_a_1_3_3')

    with tf.name_scope('reduce_a_concat') as scope:
        concat5 = tf.concat(
            [pool_a_1_1, pool_a_1_2, pool_a_1_3],
            axis=3,
            name='concat_reduce_a'
        )

    # Block B Branch 1
    with tf.name_scope('block_b_branch_1') as scope:
        pool_b_1_1 = tf.layers.max_pooling2d(
            concat5,  # Input
            pool_size=(2, 2),  # Pool size: 3x3
            strides=(1, 1),  # Stride: 2
            padding='SAME',  # "same" padding
            name='poolb_1_1'
        )

        pool_b_1_1 = tf.layers.conv2d(
            pool_b_1_1,  # Input data
            filters=128,  # 32 filters
            kernel_size=(1, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=927),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='poolb_1_1'
        )

        pool_b_1_1 = tf.layers.batch_normalization(
            pool_b_1_1,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_pool_b_1_1'
        )

        # apply relu
        pool_b_1_1 = tf.nn.relu(pool_b_1_1, name='relu_pool_b_1_1')

    # Block B branch 2
    with tf.name_scope('block_b_branch_2') as scope:
        conv_b_2 = tf.layers.conv2d(
            concat5,  # Input data
            filters=384,  # 32 filters
            kernel_size=(1, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=927),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_b_2'
        )

        conv_b_2 = tf.layers.batch_normalization(
            conv_b_2,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_conv_b_2'
        )

        # apply relu
        conv_b_2 = tf.nn.relu(conv_b_2, name='relu_conv_b_2')

    # Block B branch 3
    with tf.name_scope('block_b_branch_3') as scope:
        ## Downsize layers
        conv_b_3 = tf.layers.conv2d(
            concat5,  # Input data
            filters=192,  # 32 filters
            kernel_size=(1, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=929),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_b_3_1'
        )

        conv_b_3 = tf.layers.batch_normalization(
            conv_b_3,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_conv_b_3_1'
        )

        # apply relu
        conv_b_3 = tf.nn.relu(conv_b_3, name='relu_conv_b_3_1')

        ## 1x7 conv
        conv_b_3 = tf.layers.conv2d(
            conv_b_3,  # Input data
            filters=224,  # 32 filters
            kernel_size=(1, 7),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=930),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_b_3_2'
        )

        conv_b_3 = tf.layers.batch_normalization(
            conv_b_3,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_conv_b_3_2'
        )

        # apply relu
        conv_b_3 = tf.nn.relu(conv_b_3, name='relu_conv_b_3_2')

        ## 7x1 conv
        conv_b_3 = tf.layers.conv2d(
            conv_b_3,  # Input data
            filters=256,  # 32 filters
            kernel_size=(7, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=931),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_b_3_3'
        )

        conv_b_3 = tf.layers.batch_normalization(
            conv_b_3,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_conv_b_3_3'
        )

        # apply relu
        conv_b_3 = tf.nn.relu(conv_b_3, name='relu_conv_b_3_3')

    # Block B branch 4
    with tf.name_scope('block_b_branch_4') as scope:
        ## Downsize layers
        conv_b_4 = tf.layers.conv2d(
            concat5,  # Input data
            filters=192,  # 32 filters
            kernel_size=(1, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=932),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_b_4_1'
        )

        conv_b_4 = tf.layers.batch_normalization(
            conv_b_4,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_conv_b_4_1'
        )

        # apply relu
        conv_b_4 = tf.nn.relu(conv_b_4, name='relu_conv_b_4_1')

        ## 1x7 conv
        conv_b_4 = tf.layers.conv2d(
            conv_b_4,  # Input data
            filters=192,  # 32 filters
            kernel_size=(1, 7),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=933),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_b_4_2'
        )

        conv_b_4 = tf.layers.batch_normalization(
            conv_b_4,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_conv_b_4_2'
        )

        # apply relu
        conv_b_4 = tf.nn.relu(conv_b_4, name='relu_conv_b_4_2')

        ## 7x1 conv
        conv_b_4 = tf.layers.conv2d(
            conv_b_4,  # Input data
            filters=224,  # 32 filters
            kernel_size=(7, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=934),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_b_4_3'
        )

        conv_b_4 = tf.layers.batch_normalization(
            conv_b_4,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_conv_b_4_3'
        )

        # apply relu
        conv_b_4 = tf.nn.relu(conv_b_4, name='relu_conv_b_4_3')

        ## 1x7 conv 2
        conv_b_4 = tf.layers.conv2d(
            conv_b_4,  # Input data
            filters=224,  # 32 filters
            kernel_size=(1, 7),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=935),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_b_4_4'
        )

        conv_b_4 = tf.layers.batch_normalization(
            conv_b_4,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_conv_b_4_4'
        )

        # apply relu
        conv_b_4 = tf.nn.relu(conv_b_4, name='relu_conv_b_4_4')

        ## 7x1 conv 2
        conv_b_4 = tf.layers.conv2d(
            conv_b_4,  # Input data
            filters=256,  # 32 filters
            kernel_size=(7, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=936),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_b_4_5'
        )

        conv_b_4 = tf.layers.batch_normalization(
            conv_b_4,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_conv_b_4_5'
        )

        # apply relu
        conv_b_4 = tf.nn.relu(conv_b_4, name='relu_conv_b_4_5')

    # concatenate 6
    with tf.name_scope("concat6") as scope:
        concat6 = tf.concat(
            [pool_b_1_1, conv_b_2, conv_b_3, conv_b_4],
            axis=3,
            name='concat6'
        )

    with tf.name_scope('reduce_b_1') as scope:
        ## normal max pool
        pool_b_1_1 = tf.layers.max_pooling2d(
            concat6,  # Input
            pool_size=(3, 3),  # Pool size: 3x3
            strides=(2, 2),  # Stride: 2
            padding='SAME',  # "same" padding
            name='poolb_1_1'
        )

    with tf.name_scope('reduce_b_2') as scope:
        ## conv with stride 2
        pool_b_1_2 = tf.layers.conv2d(
            concat6,  # Input data
            filters=128,  # 32 filters
            kernel_size=(3, 3),  # Kernel size: 9x9
            strides=(2, 2),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=937),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='pool_b_1_2'
        )

        pool_b_1_2 = tf.layers.batch_normalization(
            pool_b_1_2,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_pool_b_1_2'
        )

        # apply relu
        pool_b_1_2 = tf.nn.relu(pool_b_1_2, name='relu_pool_b_1_2')

    with tf.name_scope('reduce_b_3') as scope:
        ## multiple convs
        pool_b_1_3 = tf.layers.conv2d(
            concat6,  # Input data
            filters=96,  # 32 filters
            kernel_size=(1, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=938),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='pool_b_1_3'
        )

        pool_b_1_3 = tf.layers.batch_normalization(
            pool_b_1_3,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_pool_b_1_3'
        )

        # apply relu
        pool_b_1_3 = tf.nn.relu(pool_a_1_3, name='relu_pool_b_1_3')

        pool_b_1_3 = tf.layers.conv2d(
            pool_b_1_3,  # Input data
            filters=128,  # 32 filters
            kernel_size=(3, 3),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=939),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='pool_b_1_3_2'
        )

        pool_b_1_3 = tf.layers.batch_normalization(
            pool_b_1_3,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_pool_b_1_3_2'
        )

        # apply relu
        pool_b_1_3 = tf.nn.relu(pool_b_1_3, name='relu_pool_b_1_3_2')

        pool_b_1_3 = tf.layers.conv2d(
            pool_b_1_3,  # Input data
            filters=192,  # 32 filters
            kernel_size=(3, 3),  # Kernel size: 9x9
            strides=(2, 2),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=940),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='pool_b_1_3_3'
        )

        pool_b_1_3 = tf.layers.batch_normalization(
            pool_b_1_3,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_pool_b_1_3_3'
        )

        # apply relu
        pool_b_1_3 = tf.nn.relu(pool_b_1_3, name='relu_pool_b_1_3_3')

    with tf.name_scope('reduce_b_concat') as scope:
        concat7 = tf.concat(
            [pool_b_1_1, pool_b_1_2, pool_b_1_3],
            axis=3,
            name='concat_reduce_b'
        )

    # Block C Branch 1
    with tf.name_scope('block_c_1') as scope:
        conv_c_1 = tf.layers.average_pooling2d(
            concat7,  # Input
            pool_size=(2, 2),  # Pool size: 3x3
            strides=(1, 1),  # Stride: 2
            padding='SAME',  # "same" padding
            name='conv_c_1_1'
        )

        conv_c_1 = tf.layers.conv2d(
            conv_c_1,  # Input data
            filters=256,  # 32 filters
            kernel_size=(1, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=940),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_c_1_2'
        )

        conv_c_1 = tf.layers.batch_normalization(
            conv_c_1,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_conv_c_1'
        )

        # apply relu
        conv_c_1 = tf.nn.relu(conv_c_1, name='relu_conv_c_1')

    # Block C Branch 2
    with tf.name_scope('block_c_2') as scope:
        conv_c_2 = tf.layers.conv2d(
            concat7,  # Input data
            filters=256,  # 32 filters
            kernel_size=(1, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=940),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_c_2_1'
        )

        conv_c_2 = tf.layers.batch_normalization(
            conv_c_2,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_conv_c_2'
        )

        # apply relu
        conv_c_2 = tf.nn.relu(conv_c_2, name='relu_conv_c_2')

    # Block C Branch 3
    with tf.name_scope('block_c_3') as scope:
        conv_c_3 = tf.layers.conv2d(
            concat7,  # Input data
            filters=384,  # 32 filters
            kernel_size=(1, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=940),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_c_3_1'
        )

        conv_c_3 = tf.layers.batch_normalization(
            conv_c_3,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_conv_c_3_1'
        )

        # apply relu
        conv_c_3 = tf.nn.relu(conv_c_3, name='relu_conv_c_3_1')

        ## 1x3 filters
        conv_c_3_1 = tf.layers.conv2d(
            conv_c_3,  # Input data
            filters=256,  # 32 filters
            kernel_size=(1, 3),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=940),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_c_3_2'
        )

        conv_c_3_1 = tf.layers.batch_normalization(
            conv_c_3_1,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_conv_c_3_2'
        )

        # apply relu
        conv_c_3_1 = tf.nn.relu(conv_c_3_1, name='relu_conv_c_3_2')

        ## 3x1 filters
        conv_c_3_2 = tf.layers.conv2d(
            conv_c_3,  # Input data
            filters=256,  # 32 filters
            kernel_size=(3, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=940),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_c_3_3'
        )

        conv_c_3_2 = tf.layers.batch_normalization(
            conv_c_3_2,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_conv_c_3_3'
        )

        # apply relu
        conv_c_3_2 = tf.nn.relu(conv_c_3_2, name='relu_conv_c_3_3')

    # Block C Branch 4
    with tf.name_scope('block_c_4') as scope:
        conv_c_4 = tf.layers.conv2d(
            concat7,  # Input data
            filters=384,  # 32 filters
            kernel_size=(1, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=940),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_c_4_1'
        )

        conv_c_4 = tf.layers.batch_normalization(
            conv_c_4,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_conv_c_4_1'
        )

        # apply relu
        conv_c_4 = tf.nn.relu(conv_c_4, name='relu_conv_c_4_1')

        conv_c_4 = tf.layers.conv2d(
            conv_c_4,  # Input data
            filters=448,  # 32 filters
            kernel_size=(1, 3),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=940),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_c_4_2'
        )

        conv_c_4 = tf.layers.batch_normalization(
            conv_c_4,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_conv_c_4_2'
        )

        # apply relu
        conv_c_4 = tf.nn.relu(conv_c_4, name='relu_conv_c_4_2')

        conv_c_4 = tf.layers.conv2d(
            conv_c_4,  # Input data
            filters=512,  # 32 filters
            kernel_size=(3, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=940),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_c_4_3'
        )

        conv_c_4 = tf.layers.batch_normalization(
            conv_c_4,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_conv_c_4_3'
        )

        # apply relu
        conv_c_4 = tf.nn.relu(conv_c_4, name='relu_conv_c_4_3')

        ## 1x3 filters
        conv_c_4_1 = tf.layers.conv2d(
            conv_c_4,  # Input data
            filters=256,  # 32 filters
            kernel_size=(1, 3),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=940),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_c_4_4_1'
        )

        conv_c_4_1 = tf.layers.batch_normalization(
            conv_c_4_1,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_conv_c_4_4_1'
        )

        # apply relu
        conv_c_4_1 = tf.nn.relu(conv_c_4_1, name='relu_conv_c_4_4_1')

        ## 3x1 filters
        conv_c_4_2 = tf.layers.conv2d(
            conv_c_4,  # Input data
            filters=256,  # 32 filters
            kernel_size=(3, 1),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=940),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv_c_4_4_2'
        )

        conv_c_4_2 = tf.layers.batch_normalization(
            conv_c_4_2,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_conv_c_4_4_2'
        )

        # apply relu
        conv_c_4_2 = tf.nn.relu(conv_c_4_2, name='relu_conv_c_4_4_2')

    with tf.name_scope('concat8') as scope:
        concat8 = tf.concat(
            [conv_c_1, conv_c_3_1, conv_c_3_2, conv_c_4_1, conv_c_4_2],
            axis=3,
            name='concat_8'
        )
    # Max pooling layer 2
    with tf.name_scope('pool2') as scope:
        ## Average Pooling
        pool2 = tf.layers.average_pooling2d(
            concat8,  # Input
            pool_size=(2, 2),  # Pool size: 3x3
            strides=(2, 2),  # Stride: 2
            padding='SAME',  # "same" padding
            name='pool1'
        )

    # Flatten output
    with tf.name_scope('flatten') as scope:
        #flat_output = tf.contrib.layers.flatten(pool2)
        flat_output = tf.reduce_mean(pool2, axis=[1, 2])

        # dropout at fc rate
        flat_output = tf.layers.dropout(flat_output, rate=0.2, seed=2300, training=training)

    # Fully connected layer 1
    with tf.name_scope('fc1') as scope:
        fc1 = tf.layers.dense(
            flat_output,
            2048,
            activation=None,
            kernel_initializer=tf.variance_scaling_initializer(scale=2, seed=2400),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamF),
            name="fc1"
        )

        bn_fc1 = tf.layers.batch_normalization(
            fc1,
            axis=-1,
            momentum=0.9,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_fc1'
        )

        fc1_relu = tf.nn.relu(bn_fc1, name='fc1_relu')

        # dropout
        fc1_relu = tf.layers.dropout(fc1_relu, rate=fcdropout_rate, seed=2500, training=training)

    # Fully connected layer 2
    with tf.name_scope('fc2') as scope:
        fc2 = tf.layers.dense(
            fc1_relu,  # input
            1024,  # 1024 hidden units
            activation=None,  # None
            kernel_initializer=tf.variance_scaling_initializer(scale=2, seed=2600),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamF),
            name="fc2"
        )

        bn_fc2 = tf.layers.batch_normalization(
            fc2,
            axis=-1,
            momentum=0.9,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_fc2'
        )

        fc2_relu = tf.nn.relu(bn_fc2, name='fc2_relu')

        # dropout
        fc2_relu = tf.layers.dropout(fc2_relu, rate=fcdropout_rate, seed=2700, training=training)

    # Output layer
    logits = tf.layers.dense(
        fc2_relu,
        num_classes,      # One output unit per category
        activation=None,  # No activation function
        kernel_initializer=tf.variance_scaling_initializer(scale=1, seed=2800),
        bias_initializer=tf.zeros_initializer(),
        name="logits"
    )

    with tf.variable_scope('conv1.0', reuse=True):
        conv_kernels1 = tf.get_variable('kernel')
        kernel_transposed = tf.transpose(conv_kernels1, [3, 0, 1, 2])

    with tf.variable_scope('visualization'):
        tf.summary.image('conv1.0/filters', kernel_transposed, max_outputs=32)

    ## Loss function options
    # Regular mean cross entropy
    mean_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

    # weighted mean cross entropy
    # onehot_labels = tf.one_hot(y, depth=num_classes)
    # mean_ce = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=tf.one_hot(y, depth=num_classes), logits=logits, pos_weight=classes_weights))

    # Different weighting method
    # This will weight the positive examples higher so as to improve recall
    #weights = tf.multiply(2, tf.cast(tf.equal(y, 1), tf.int32)) + 1
    #mean_ce = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits, weights=weights))

    # Add in l2 loss
    loss = mean_ce + tf.losses.get_regularization_loss()

    # Adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Minimize cross-entropy
    train_op = optimizer.minimize(loss, global_step=global_step)

    # Compute predictions and accuracy
    predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
    is_correct = tf.equal(y, predictions)
    accuracy = tf.reduce_mean(tf.cast(is_correct, dtype=tf.float32))

    # calculate recall
    if num_classes > 2:
        recall = [0] * num_classes
        rec_op = [[]] * num_classes

        for k in range(num_classes):
            recall[k], rec_op[k] = tf.metrics.recall(
                labels=tf.equal(y, k),
                predictions=tf.equal(predictions, k)
            )
    else:
        recall, rec_op = tf.metrics.recall(labels=y, predictions=predictions, name="recall")
        precision, prec_op = tf.metrics.precision(labels=y, predictions=predictions, name="precision")
        f1_score = 2 * ( (precision * recall) / (precision + recall))

    # add this so that the batch norm gets run
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Create summary hooks
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('recall_1', recall)
    tf.summary.scalar('cross_entropy', mean_ce)

    if num_classes == 2:
        tf.summary.scalar('precision_1', precision)
        tf.summary.scalar('f1_score', f1_score)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('learning_rate', learning_rate)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()

    print("Graph created...")
# ## Train

## CONFIGURE OPTIONS
# if a checkpoint exists for the model restore it, otherwise initialize a new one
if os.path.exists(os.path.join("model", model_name + '.ckpt.index')):
    init = False
else:
    init = True

crop = False                  # do random cropping of images?

meta_data_every = 1
log_to_tensorboard = True
print_every = 3                # how often to print metrics
checkpoint_every = 1           # how often to save model in epochs
use_gpu = False                 # whether or not to use the GPU
print_metrics = True          # whether to print or plot metrics, if False a plot will be created and updated every epoch
evaluate = True               # whether to periodically evaluate on test data

# Placeholders for metrics
if init:
    valid_acc_values = []
    valid_recall_values = []
    valid_cost_values = []
    train_acc_values = []
    train_recall_values = []
    train_cost_values = []
    train_lr_values = []
    train_loss_values = []
    
config = tf.ConfigProto()
#if use_gpu:
#    config = tf.ConfigProto()
#    config.gpu_options.allocator_type = 'BFC'
#    config.gpu_options.per_process_gpu_memory_fraction = 0.7
#else:
#    config = tf.ConfigProto(device_count = {'GPU': 0})

## train the model
with tf.Session(graph=graph, config=config) as sess:
    if log_to_tensorboard:
        train_writer = tf.summary.FileWriter('./logs/tr_' + model_name, sess.graph)
        test_writer = tf.summary.FileWriter('./logs/te_' + model_name)
    
    if not print_metrics:
        # create a plot to be updated as model is trained
        f, ax = plt.subplots(1,4,figsize=(24,5))
    
    # create the saver
    saver = tf.train.Saver()
    
    # If the model is new initialize variables, else restore the session
    if init:
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, './model/'+model_name+'.ckpt')

    sess.run(tf.local_variables_initializer())
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print("Training model", model_name,"...")
    
    for epoch in range(epochs):
        for i in range(steps_per_epoch):
            # Accuracy values (train) after each batch
            batch_acc = []
            batch_cost = []
            batch_loss = []
            batch_lr = []
            batch_recall = []

            # create the metadata
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            # Run training and evaluate accuracy
            _, _, precision_value, summary, acc_value, cost_value, loss_value, recall_value, step, lr = sess.run([train_op, extra_update_ops, prec_op,
                     merged, accuracy, mean_ce, loss, rec_op, global_step,
                     learning_rate], feed_dict={
                        #X: X_batch,
                        #y: y_batch,
                        training: True,
                        is_testing: False,
                    },
            options=run_options,
            run_metadata=run_metadata)

            # Save accuracy (current batch)
            batch_acc.append(acc_value)
            batch_cost.append(cost_value)
            batch_lr.append(lr)
            batch_loss.append(loss_value)
            batch_recall.append(np.mean(recall_value))
            
            # write the summary
            if log_to_tensorboard:
                train_writer.add_summary(summary, step)
                # only log the meta data once per epoch
                if i == (steps_per_epoch - 1):
                    train_writer.add_run_metadata(run_metadata, 'step %d' % step)
                
        # save checkpoint every nth epoch
        if(epoch % checkpoint_every == 0):
            print("Saving checkpoint")
            save_path = saver.save(sess, './model/'+model_name+'.ckpt')
    
            # Now that model is saved set init to false so we reload it next time
            init = False
        
        # init batch arrays
        batch_cv_acc = []
        batch_cv_cost = []
        batch_cv_loss = []
        batch_cv_recall = []
        batch_cv_precision = []

        ## evaluate on test data if it exists, otherwise ignore this step
        if evaluate:
            print("Evaluating model...")
            # load the test data
            X_cv, y_cv = load_validation_data(percentage=1, how="normal")

            # evaluate the test data
            for X_batch, y_batch in get_batches(X_cv, y_cv, batch_size // 2, distort=False):
                summary, valid_acc, valid_recall, valid_precision, valid_cost, valid_loss = sess.run(
                    [merged, accuracy, rec_op, prec_op, mean_ce, loss],
                    feed_dict={
                        X: X_batch,
                        y: y_batch,
                        is_testing: True,
                        training: False
                    })

                batch_cv_acc.append(valid_acc)
                batch_cv_cost.append(valid_cost)
                batch_cv_loss.append(valid_loss)
                batch_cv_recall.append(np.mean(valid_recall))
                batch_cv_precision.append(np.mean(valid_precision))

            # Write average of validation data to summary logs
            if log_to_tensorboard:
                summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=np.mean(batch_cv_acc)),
                                            tf.Summary.Value(tag="cross_entropy", simple_value=np.mean(batch_cv_cost)),
                                            tf.Summary.Value(tag="recall_1", simple_value=np.mean(batch_cv_recall)),
                                            tf.Summary.Value(tag="precision_1",
                                                             simple_value=np.mean(batch_cv_precision)), ])
                test_writer.add_summary(summary, step)
                step += 1

            # delete the test data to save memory
            del (X_cv)
            del (y_cv)

            print("Done evaluating...")
        else:
            batch_cv_acc.append(0)
            batch_cv_cost.append(0)
            batch_cv_loss.append(0)
            batch_cv_recall.append(0)
          
        # take the mean of the values to add to the metrics
        valid_acc_values.append(np.mean(batch_cv_acc))
        valid_cost_values.append(np.mean(batch_cv_cost))
        train_acc_values.append(np.mean(batch_acc))
        train_cost_values.append(np.mean(batch_cost))
        train_lr_values.append(np.mean(batch_lr))
        train_loss_values.append(np.mean(batch_loss))
        train_recall_values.append(np.mean(batch_recall))
        valid_recall_values.append(np.mean(batch_cv_recall))

        # Print progress every nth epoch to keep output to reasonable amount
        if(epoch % print_every == 0):
            print('Epoch {:02d} - step {} - cv acc: {:.3f} - train acc: {:.3f} (mean) - cv cost: {:.3f} - lr: {:.5f}'.format(
                epoch, step, np.mean(batch_cv_acc), np.mean(batch_acc), np.mean(batch_cv_cost), lr
            ))

        # Print data every 50th epoch so I can write it down to compare models
        if (not print_metrics) and (epoch % 50 == 0) and (epoch > 1):
            if(epoch % print_every == 0):
                print('Epoch {:02d} - step {} - cv acc: {:.4f} - train acc: {:.3f} (mean) - cv cost: {:.3f} - lr: {:.5f}'.format(
                    epoch, step, np.mean(batch_cv_acc), np.mean(batch_acc), np.mean(batch_cv_cost), lr
                ))  
      
    # stop the coordinator
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)

