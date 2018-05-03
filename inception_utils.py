import numpy as np
import tensorflow as tf
from training_utils import _conv2d_batch_norm, _dense_batch_norm

def _stem(input, lamC=0.0, training = tf.placeholder(dtype=tf.bool, name="is_training")):
    conv1 = _conv2d_batch_norm(input, 32, kernel_size=(3, 3), stride=(2, 2), training=training, epsilon=1e-8,
                               padding="VALID", seed=100, lambd=lamC, name="stem_1.1")

    conv1 = _conv2d_batch_norm(conv1, 32, kernel_size=(3, 3), stride=(1, 1), training=training, epsilon=1e-8,
                               padding="VALID", lambd=lamC, name="stem_1.2")

    conv1 = _conv2d_batch_norm(conv1, 64, kernel_size=(3, 3), stride=(1, 1), training=training, epsilon=1e-8,
                               padding="SAME", lambd=lamC, name="stem_1.3")

    # Stem Reduce 1
    with tf.name_scope('stem_pool1') as scope:
        pool1 = tf.layers.max_pooling2d(
            conv1,  # Input
            pool_size=(3, 3),  # Pool size: 2x2
            strides=(2, 2),  # Stride: 2
            padding='VALID',  # "same" padding
            name='stem_pool1'
        )

    conv1 = _conv2d_batch_norm(conv1, 32, kernel_size=(3, 3), stride=(2, 2), training=training, epsilon=1e-8,
                               padding="VALID", lambd=lamC, name="stem_pool1.1")

    # concat 1
    with tf.name_scope("stem_concat1") as scope:
        concat1 = tf.concat(
            [pool1, conv1],
            axis=3,
            name='stem_concat1'
        )

    # Stem branch 1
    conv11 = _conv2d_batch_norm(concat1, 48, kernel_size=(1, 1), stride=(1, 1), training=training, epsilon=1e-8,
                               padding="SAME", lambd=lamC, name="stem_1.1.1")

    conv11 = _conv2d_batch_norm(conv11, 64, kernel_size=(3, 3), stride=(1, 1), training=training, epsilon=1e-8,
                                padding="VALID", lambd=lamC, name="stem_1.1.2")

    # Stem Branch 2
    conv12 = _conv2d_batch_norm(concat1, 48, kernel_size=(1, 1), stride=(1, 1), training=training, epsilon=1e-8,
                                padding="SAME", lambd=lamC, name="stem_1.2.1")

    conv12 = _conv2d_batch_norm(conv12, 64, kernel_size=(7, 1), stride=(1, 1), training=training, epsilon=1e-8,
                                padding="SAME", lambd=lamC, name="stem_1.2.2")

    conv12 = _conv2d_batch_norm(conv12, 64, kernel_size=(1, 7), stride=(1, 1), training=training, epsilon=1e-8,
                                padding="SAME", lambd=lamC, name="stem_1.2.3")

    conv12 = _conv2d_batch_norm(conv12, 64, kernel_size=(3, 3), stride=(1, 1), training=training, epsilon=1e-8,
                                padding="VALID", lambd=lamC, name="stem_1.2.4")

    # concat 2
    with tf.name_scope("stem_concat2") as scope:
        concat2 = tf.concat(
            [conv11, conv12],
            axis=3,
            name='stem_concat2'
        )

    # Stem Reduce 2
    with tf.name_scope('stem_pool2') as scope:
        pool2 = tf.layers.max_pooling2d(
            concat2,  # Input
            pool_size=(3, 3),  # Pool size: 2x2
            strides=(2, 2),  # Stride: 2
            padding='VALID',  # "same" padding
            name='stem_pool2'
        )

    conv13 = _conv2d_batch_norm(concat2, 128, kernel_size=(3, 3), stride=(2, 2), training=training, epsilon=1e-8,
                               padding="VALID", lambd=lamC, name="stem_pool2.1")

    # concat 3
    with tf.name_scope("stem_concat3") as scope:
        concat3 = tf.concat(
            [pool2, conv13],
            axis=3,
            name='stem_concat3'
        )

    return concat3


def _block_a(input, name, lamC=0.0, training = tf.placeholder(dtype=tf.bool, name="is_training")):
    ## Branch 1 - average pool and 1x1 conv
    with tf.name_scope(name+"a_branch_1_pool") as scope:
        branch1 = tf.layers.average_pooling2d(
            input,  # Input
            pool_size=(2, 2),  # Pool size: 2x2
            strides=(1, 1),  # Stride: 2
            padding='SAME',  # "same" padding
            name=name+"a_branch_1_pool"
        )

    branch1 = _conv2d_batch_norm(branch1, 48, kernel_size=(1, 1), stride=(1, 1), training=training, epsilon=1e-8,
                            padding="SAME", lambd=lamC, name=name+"a_branch_1_conv_1.0")

    ## Branch 2 - 1x1 conv
    branch2 = _conv2d_batch_norm(input, 64, kernel_size=(1, 1), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "a_branch_2_conv_1.0")

    ## Branch 3
    branch3 = _conv2d_batch_norm(input, 48, kernel_size=(1, 1), stride=(1, 1), training=training, epsilon=1e-8,
                             padding="SAME", lambd=lamC, name=name + "a_branch_3_conv_1.0")

    branch3 = _conv2d_batch_norm(branch3, 64, kernel_size=(3, 3), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "a_branch_3_conv_1.1")

    ## Branch 4
    branch4 = _conv2d_batch_norm(input, 48, kernel_size=(1, 1), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "a_branch_4_conv_1.0")

    branch4 = _conv2d_batch_norm(branch4, 64, kernel_size=(3, 3), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "a_branch_4_conv_1.1")

    branch4 = _conv2d_batch_norm(branch4, 64, kernel_size=(3, 3), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "a_branch_4_conv_1.2")

    # concat 1
    with tf.name_scope(name+"a_concat_1") as scope:
        concat1 = tf.concat(
            [branch1, branch2, branch3, branch4],
            axis=3,
            name=name+'a_concat_1'
        )

    return concat1


def _block_b(input, name, lamC=0.0, training = tf.placeholder(dtype=tf.bool, name="is_training")):
    ## Branch 1 - average pool and 1x1 conv
    with tf.name_scope(name+"b_branch_1_pool") as scope:
        branch1 = tf.layers.average_pooling2d(
            input,  # Input
            pool_size=(2, 2),  # Pool size: 2x2
            strides=(1, 1),  # Stride: 2
            padding='SAME',  # "same" padding
            name=name+"b_branch_1_pool"
        )

    branch1 = _conv2d_batch_norm(branch1, 96, kernel_size=(1, 1), stride=(1, 1), training=training, epsilon=1e-8,
                            padding="SAME", lambd=lamC, name=name+"b_branch_1_conv_1.0")

    ## Branch 2 - 1x1 conv
    branch2 = _conv2d_batch_norm(input, 128, kernel_size=(1, 1), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "b_branch_2_conv_1.0")

    ## Branch 3
    branch3 = _conv2d_batch_norm(input, 96, kernel_size=(1, 1), stride=(1, 1), training=training, epsilon=1e-8,
                             padding="SAME", lambd=lamC, name=name + "b_branch_3_conv_1.0")

    branch3 = _conv2d_batch_norm(branch3, 128, kernel_size=(1, 7), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "b_branch_3_conv_1.1")

    branch3 = _conv2d_batch_norm(branch3, 192, kernel_size=(7, 1), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "b_branch_3_conv_1.2")

    ## Branch 4
    branch4 = _conv2d_batch_norm(input, 96, kernel_size=(1, 1), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "b_branch_4_conv_1.0")

    branch4 = _conv2d_batch_norm(branch4, 96, kernel_size=(1, 7), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "b_branch_4_conv_1.1")

    branch4 = _conv2d_batch_norm(branch4, 128, kernel_size=(7, 1), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "b_branch_4_conv_1.2")

    branch4 = _conv2d_batch_norm(branch4, 128, kernel_size=(1, 7), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "b_branch_4_conv_1.3")

    branch4 = _conv2d_batch_norm(branch4, 192, kernel_size=(7, 1), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "b_branch_4_conv_1.4")

    # concat 1
    with tf.name_scope(name+"b_concat_1") as scope:
        concat1 = tf.concat(
            [branch1, branch2, branch3, branch4],
            axis=3,
            name=name+'b_concat_1'
        )

    return concat1

def _block_c(input, name, lamC=0.0, training = tf.placeholder(dtype=tf.bool, name="is_training")):
    ## Branch 1 - average pool and 1x1 conv
    with tf.name_scope(name+"b_branch_1_pool") as scope:
        branch1 = tf.layers.average_pooling2d(
            input,  # Input
            pool_size=(2, 2),  # Pool size: 2x2
            strides=(1, 1),  # Stride: 2
            padding='SAME',  # "same" padding
            name=name+"c_branch_1_pool"
        )

    branch1 = _conv2d_batch_norm(branch1, 128, kernel_size=(1, 1), stride=(1, 1), training=training, epsilon=1e-8,
                            padding="SAME", lambd=lamC, name=name+"c_branch_1_conv_1.0")

    ## Branch 2 - 1x1 conv
    branch2 = _conv2d_batch_norm(input, 128, kernel_size=(1, 1), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "c_branch_2_conv_1.0")

    ## Branch 3
    branch3 = _conv2d_batch_norm(input, 192, kernel_size=(1, 1), stride=(1, 1), training=training, epsilon=1e-8,
                             padding="SAME", lambd=lamC, name=name + "c_branch_3_conv_1.0")

    branch31 = _conv2d_batch_norm(branch3, 128, kernel_size=(1, 3), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "c_branch_3_conv_1.1")

    branch32 = _conv2d_batch_norm(branch3, 128, kernel_size=(3, 1), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "c_branch_3_conv_1.2")

    ## Branch 4
    branch4 = _conv2d_batch_norm(input, 192, kernel_size=(1, 1), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "c_branch_4_conv_1.0")

    branch4 = _conv2d_batch_norm(branch4, 256, kernel_size=(1, 3), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "c_branch_4_conv_1.1")

    branch4 = _conv2d_batch_norm(branch4, 256, kernel_size=(3, 1), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "c_branch_4_conv_1.2")

    branch41 = _conv2d_batch_norm(branch4, 128, kernel_size=(1, 3), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "c_branch_4_conv_1.3")

    branch42 = _conv2d_batch_norm(branch4, 128, kernel_size=(3, 1), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "c_branch_4_conv_1.4")

    # concat 1
    with tf.name_scope(name+"b_concat_1") as scope:
        concat1 = tf.concat(
            [branch1, branch2, branch31, branch32, branch41, branch42],
            axis=3,
            name=name+'b_concat_1'
        )

    return concat1

def _reduce_a(input, name, k, l, m, n, training = tf.placeholder(dtype=tf.bool, name="is_training"), lamC=0.0):
    # branch 1
    with tf.name_scope(name+"reduce_a_branch_1") as scope:
        branch1 = tf.layers.max_pooling2d(
            input,  # Input
            pool_size=(3, 3),  # Pool size: 2x2
            strides=(2, 2),  # Stride: 2
            padding='VALID',  # "same" padding
            name=name+'a_reduce_branch_1'
        )

    # branch 2
    branch2 = _conv2d_batch_norm(input, n, kernel_size=(3, 3), stride=(2, 2), training=training, epsilon=1e-8,
                               padding="VALID", lambd=lamC, name=name+"a_reduce_branch_2")

    # Branch 3
    branch3 = _conv2d_batch_norm(input, k, kernel_size=(1, 1), stride=(1, 1), training=training, epsilon=1e-8,
                               padding="SAME", lambd=lamC, name=name+"a_reduce_branch_3.1")

    branch3 = _conv2d_batch_norm(branch3, l, kernel_size=(3, 3), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "a_reduce_branch_3.2")

    branch3 = _conv2d_batch_norm(branch3, m, kernel_size=(3, 3), stride=(2, 2), training=training, epsilon=1e-8,
                                 padding="VALID", lambd=lamC, name=name + "a_reduce_branch_3.3")

    # concat 1
    with tf.name_scope(name + "a_reduce_concat_1") as scope:
        concat1 = tf.concat(
            [branch1, branch2, branch3],
            axis=3,
            name=name + 'a_reduce_concat'
        )

    return concat1

def _reduce_b(input, name, training = tf.placeholder(dtype=tf.bool, name="is_training"), lamC=0.0):
    # branch 1
    with tf.name_scope(name+"reduce_b_branch_1") as scope:
        branch1 = tf.layers.max_pooling2d(
            input,  # Input
            pool_size=(3, 3),  # Pool size: 2x2
            strides=(2, 2),  # Stride: 2
            padding='VALID',  # "same" padding
            name=name+'b_reduce_branch_1'
        )

    # branch 2
    branch2 = _conv2d_batch_norm(input, 96, kernel_size=(1, 1), stride=(1, 1), training=training, epsilon=1e-8,
                               padding="SAME", lambd=lamC, name=name+"b_reduce_branch_2")

    branch2 = _conv2d_batch_norm(branch2, 96, kernel_size=(3, 3), stride=(2, 2), training=training, epsilon=1e-8,
                                 padding="VALID", lambd=lamC, name=name + "b_reduce_branch_2.1")

    # Branch 3
    branch3 = _conv2d_batch_norm(input, 128, kernel_size=(1, 1), stride=(1, 1), training=training, epsilon=1e-8,
                               padding="SAME", lambd=lamC, name=name+"b_reduce_branch_3.1")

    branch3 = _conv2d_batch_norm(branch3, 128, kernel_size=(1, 7), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "b_reduce_branch_3.2")

    branch3 = _conv2d_batch_norm(branch3, 192, kernel_size=(7, 1), stride=(1, 1), training=training, epsilon=1e-8,
                                 padding="SAME", lambd=lamC, name=name + "b_reduce_branch_3.3")

    branch3 = _conv2d_batch_norm(branch3, 192, kernel_size=(3, 3), stride=(2, 2), training=training, epsilon=1e-8,
                                 padding="VALID", lambd=lamC, name=name + "b_reduce_branch_3.4")
    # concat 1
    with tf.name_scope(name + "b_reduce_concat_1") as scope:
        concat1 = tf.concat(
            [branch1, branch2, branch3],
            axis=3,
            name=name + 'b_reduce_concat'
        )

    return concat1