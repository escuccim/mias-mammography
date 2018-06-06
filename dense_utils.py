import tensorflow as tf

def _dense_block(input, layers, growth_rate=12, bottleneck=False, training=tf.placeholder(dtype=tf.bool, name="is_training"), seed=None, name=None, activation="relu"):
    # with tf.name_scope('block_' + name) as scope:
    # input layer
    layer1 = _dense_layer(input, growth_rate, training=training, name=name + "_layer1")

    layer2 = _dense_layer(layer1, growth_rate, training=training, name=name + "_layer2")

    concat_inputs = [layer1, layer2]

    for i in range(2, layers):
        # concat the inputs
        layer_inputs = tf.concat(concat_inputs,
                                axis=3,
                                name="concat_"+name+"_" + str(i))

        # optional bottleneck
        # if bottleneck:
        #     layer_inputs = _bottleneck(input, growth_rate, training = training, name=name + "_bottleneck_" + str(i))

        layers = _dense_layer(layer_inputs, growth_rate, training=training, name=name + "_layer_" + str(i))

        concat_inputs.append(layers)

    output = tf.concat(concat_inputs,
                                axis=3,
                                name="concat_"+name+"_final")

    return output

def _dense_layer(input, filters, stride=(1,1), training = tf.placeholder(dtype=tf.bool, name="is_training"), epsilon=1e-8, padding="SAME", seed=None, lambd=0.0, name=None, activation="relu"):

    with tf.name_scope('dense_'+name) as scope:
        # batch norm
        layer = tf.layers.batch_normalization(
                input,
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
                name='bn_'+name
            )

        # activation
        if activation == "relu":
            # apply relu
            layer = tf.nn.relu(layer, name='relu_' + name)
        elif activation == "elu":
            layer = tf.nn.elu(layer, name="elu_" + name)

        # 3x3 convolution
        layer = tf.layers.conv2d(
                layer,
                filters=filters,
                kernel_size=(3,3),
                strides=stride,
                padding=padding,
                activation=None,
                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=seed),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambd),
                name='conv_'+name
            )

    return layer

def _transition(input, filters, training = tf.placeholder(dtype=tf.bool, name="is_training"), epsilon=1e-8, padding="SAME", seed=None, lambd=0.0, name=None, activation="relu"):
    with tf.name_scope('transition_' + name) as scope:
        # batch norm
        layer = tf.layers.batch_normalization(
            input,
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
            name='tn_bn_' + name
        )

        # 1x1 conv
        layer = tf.layers.conv2d(
            layer,
            filters=filters,
            kernel_size=(1, 1),
            strides=(1,1),
            padding=padding,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=seed),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambd),
            name='1x1conv_' + name
        )

        # average pooling
        layer = tf.layers.average_pooling2d(
            layer,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='SAME',
            name='pool_' + name
        )

    return layer

def _bottleneck(input, growth_rate, training = tf.placeholder(dtype=tf.bool, name="is_training"), epsilon=1e-8, padding="SAME", seed=None, lambd=0.0, name=None, activation="relu"):
    with tf.name_scope('bottleneck_' + name) as scope:
        # batch norm
        layer = tf.layers.batch_normalization(
                input,
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
                name='bottleneck_bn_'+name
            )

        # activation
        if activation == "relu":
            # apply relu
            layer = tf.nn.relu(layer, name='bottleneck_relu_' + name)
        elif activation == "elu":
            layer = tf.nn.elu(layer, name="bottleneck_elu_" + name)

        # 1x1 convolution
        layer = tf.layers.conv2d(
                layer,
                filters=growth_rate * 4,
                kernel_size=(1,1),
                strides=(1,1),
                padding=padding,
                activation=None,
                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=seed),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambd),
                name='bottleneck_'+name
            )

    return layer