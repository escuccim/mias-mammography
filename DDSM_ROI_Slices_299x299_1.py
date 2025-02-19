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
epochs_per_decay = 25
starting_rate = 0.002
decay_factor = 0.85
staircase = True

# learning rate decay variables
steps_per_epoch = int(22539 / batch_size)
print("Steps per epoch:", steps_per_epoch)

# lambdas
lamC = 0.00000
lamF = 0.00010

# use dropout
dropout = False
fcdropout_rate = 0.5
convdropout_rate = 0.1

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
model_name = "model_s0.0.0.9"
# 0.0.0.4 - increase pool3 to 3x3 with stride 3
# 0.0.0.6 - reduce pool 3 stride back to 2
# 0.0.0.7 - reduce lambda for l2 reg
# 0.0.0.8 - increase conv1 to 7x7 stride 2
# 0.0.0.9 - disable per image normalization

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

    # Convolutional layer 1
    with tf.name_scope('conv1') as scope:
        conv1 = tf.layers.conv2d(
            X,  # Input data
            filters=32,  # 32 filters
            kernel_size=(7, 7),  # Kernel size: 5x5
            strides=(2, 2),  # Stride: 2
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=10),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv1'
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
            name='bn1'
        )

        # apply relu
        conv1_bn_relu = tf.nn.relu(conv1, name='relu1')

        if dropout:
            conv1_bn_relu = tf.layers.dropout(conv1_bn_relu, rate=convdropout_rate, seed=9, training=training)

    with tf.name_scope('conv1.1') as scope:
        conv11 = tf.layers.conv2d(
            conv1_bn_relu,  # Input data
            filters=32,  # 32 filters
            kernel_size=(5, 5),  # Kernel size: 5x5
            strides=(1, 1),  # Stride: 2
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=10),
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
        conv11_bn_relu = tf.nn.relu(conv11, name='relu1.1')

        if dropout:
            conv11_bn_relu = tf.layers.dropout(conv11_bn_relu, rate=convdropout_rate, seed=9, training=training)

    # Max pooling layer 1
    with tf.name_scope('pool1') as scope:
        pool1 = tf.layers.max_pooling2d(
            conv11_bn_relu,  # Input
            pool_size=(3, 3),  # Pool size: 3x3
            strides=(2, 2),  # Stride: 2
            padding='SAME',  # "same" padding
            name='pool1'
        )

        if dropout:
            # dropout at 10%
            pool1 = tf.layers.dropout(pool1, rate=convdropout_rate, seed=1, training=training)

    # Convolutional layer 2
    with tf.name_scope('conv2') as scope:
        conv2 = tf.layers.conv2d(
            pool1,  # Input data
            filters=64,  # 32 filters
            kernel_size=(3, 3),  # Kernel size: 9x9
            strides=(1, 1),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=10),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv2'
        )

        conv2 = tf.layers.batch_normalization(
            conv2,
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
            name='bn2'
        )

        # apply relu
        conv2_bn_relu = tf.nn.relu(conv2, name='relu2')

        if dropout:
            conv2_bn_relu = tf.layers.dropout(conv2_bn_relu, rate=convdropout_rate, seed=9, training=training)

    # Max pooling layer 2
    with tf.name_scope('pool2') as scope:
        pool2 = tf.layers.max_pooling2d(
            conv2_bn_relu,  # Input
            pool_size=(2, 2),  # Pool size: 3x3
            strides=(2, 2),  # Stride: 2
            padding='SAME',  # "same" padding
            name='pool1'
        )

        if dropout:
            # dropout at 10%
            pool2 = tf.layers.dropout(pool2, rate=convdropout_rate, seed=1, training=training)

    # Convolutional layer 3
    with tf.name_scope('conv3') as scope:
        conv3 = tf.layers.conv2d(
            pool2,  # Input data
            filters=128,  # 48 filters
            kernel_size=(3, 3),  # Kernel size: 5x5
            strides=(2, 2),  # Stride: 1
            padding='SAME',  # "same" padding
            activation=None,  # None
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=10),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv3'
        )

        bn3 = tf.layers.batch_normalization(
            conv3,
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
            name='bn3'
        )

        # apply relu
        conv3_bn_relu = tf.nn.relu(bn3, name='relu3')

        if dropout:
            conv3_bn_relu = tf.layers.dropout(conv3_bn_relu, rate=convdropout_rate, seed=9, training=training)

    # Max pooling layer 3
    with tf.name_scope('pool3') as scope:
        pool3 = tf.layers.max_pooling2d(
            conv3_bn_relu,  # Input
            pool_size=(3, 3),  # Pool size: 2x2
            strides=(2, 2),  # Stride: 2
            padding='SAME',  # "same" padding
            name='pool3'
        )

        if dropout:
            # dropout at 10%
            pool3 = tf.layers.dropout(pool3, rate=convdropout_rate, seed=1, training=training)

    # Flatten output
    with tf.name_scope('flatten') as scope:
        flat_output = tf.contrib.layers.flatten(pool3)

        # dropout at 10%
        flat_output = tf.layers.dropout(flat_output, rate=fcdropout_rate, seed=5, training=training)

    # Fully connected layer 1
    with tf.name_scope('fc1') as scope:
        fc1 = tf.layers.dense(
            flat_output,  # input
            1024,  # 2048 hidden units
            activation=None,  # None
            kernel_initializer=tf.variance_scaling_initializer(scale=2, seed=4),
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

        # dropout at 25%
        fc1_relu = tf.layers.dropout(fc1_relu, rate=fcdropout_rate, seed=10, training=training)

    # Fully connected layer 2
    with tf.name_scope('fc2') as scope:
        fc2 = tf.layers.dense(
            fc1_relu,  # input
            1024,  # 1024 hidden units
            activation=None,  # None
            kernel_initializer=tf.variance_scaling_initializer(scale=2, seed=5),
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

        # dropout at 10%
        fc2_relu = tf.layers.dropout(fc2_relu, rate=fcdropout_rate, seed=11, training=training)

    # Output layer
    logits = tf.layers.dense(
        fc2_relu,  # input
        num_classes,  # One output unit per category
        activation=None,  # No activation function
        kernel_initializer=tf.variance_scaling_initializer(scale=1, seed=6),
        bias_initializer=tf.zeros_initializer(),
        name="logits"
    )

    with tf.variable_scope('conv1', reuse=True):
        conv_kernels1 = tf.get_variable('kernel')
        kernel_transposed = tf.transpose(conv_kernels1, [3, 0, 1, 2])

    with tf.variable_scope('visualization'):
        tf.summary.image('conv1/filters', kernel_transposed, max_outputs=32)

    # This will weight the positive examples higher so as to improve recall
    weights = tf.multiply(3, tf.cast(tf.equal(y, 1), tf.int32)) + 1
    # onehot_labels = tf.one_hot(y, depth=num_classes)
    # mean_ce = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=tf.one_hot(y, depth=num_classes), logits=logits, pos_weight=classes_weights))
    # mean_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    mean_ce = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits, weights=weights))


    loss = mean_ce + tf.losses.get_regularization_loss()

    # Adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Minimize cross-entropy
    train_op = optimizer.minimize(loss, global_step=global_step)

    # Compute predictions and accuracy
    predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
    is_correct = tf.equal(y, predictions)
    accuracy = tf.reduce_mean(tf.cast(is_correct, dtype=tf.float32))

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
        # precision, prec_op = tf.metrics.precision(labels=y, predictions=predictions, name="precision")
        # f1_score = 2 * ( (precision * recall) / (precision + recall))

    # add this so that the batch norm gets run
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Create summary hooks
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('recall_1', recall)
    tf.summary.scalar('cross_entropy', mean_ce)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('learning_rate', learning_rate)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()

    print("Graph created...")
# ## Train

## CONFIGURE OPTIONS
init = True                   # whether to initialize the model or use a saved version
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
            _, _, summary, acc_value, cost_value, loss_value, recall_value, step, lr = sess.run([train_op, extra_update_ops, 
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
        
        ## evaluate on test data if it exists, otherwise ignore this step
        if evaluate:
            # load the test data
            X_cv, y_cv = load_validation_data(percentage=1, how="normal")
            
            # evaluate the test data
            for X_batch, y_batch in get_batches(X_cv, y_cv, batch_size // 2, distort=False):
                summary, valid_acc, valid_recall, valid_cost, valid_loss = sess.run([merged, accuracy, rec_op, mean_ce, loss], 
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
    
            # Write average of validation data to summary logs
            if log_to_tensorboard:
                summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=np.mean(batch_cv_acc)),
                                            tf.Summary.Value(tag="cross_entropy", simple_value=np.mean(batch_cv_cost)),
                                            tf.Summary.Value(tag="recall_1", simple_value=np.mean(batch_cv_recall)), ])
                test_writer.add_summary(summary, step)
                step += 1
            
            # delete the test data to save memory
            del(X_cv)
            del(y_cv)
        
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

