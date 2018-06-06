import numpy as np
import os
import wget
from sklearn.model_selection import train_test_split
import tensorflow as tf
from training_utils import download_file, get_batches, read_and_decode_single_example, load_validation_data, \
    download_data, evaluate_model, get_training_data, load_weights, flatten, _scale_input_data, augment, _conv2d_batch_norm, standardize
import argparse
from tensorboard import summary as summary_lib
from dense_utils import _bottleneck, _dense_block, _transition

# If number of epochs has been passed in use that, otherwise default to 50
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", help="number of epochs to train", default=30, type=int)
parser.add_argument("-d", "--data", help="which dataset to use", default=12, type=int)
parser.add_argument("-m", "--model", help="model to initialize weights with", default=None)
parser.add_argument("-r", "--restore", help="model to restore and continue training", default=None)
parser.add_argument("-l", "--label", help="how to classify data", default="mask")
parser.add_argument("-a", "--action", help="action to perform", default="train")
parser.add_argument("-f", "--freeze", help="whether to freeze convolutional layers", nargs='?', const=True, default=False)
parser.add_argument("-s", "--stop", help="stop gradient at pool5", nargs='?', const=True, default=False)
parser.add_argument("-t", "--threshold", help="decision threshold", default=0.5, type=float)
parser.add_argument("-c", "--contrast", help="contrast adjustment, if any", default=None, type=float)
parser.add_argument("-n", "--normalize", help="apply per image normalization", nargs='?', const=True, default=False)
parser.add_argument("-w", "--weight", help="weight to give to positive examples in cross-entropy", default=10, type=float)
parser.add_argument("-v", "--version", help="version or run number to assign to model name", default="")
parser.add_argument("--distort", help="use online data augmentation", default=False, const=True, nargs="?")
args = parser.parse_args()

epochs = args.epochs
dataset = args.data
init_model = args.model
restore_model = args.restore
how = args.label
action = args.action
threshold = args.threshold
freeze = args.freeze
stop = args.stop
contrast = args.contrast
normalize = args.normalize
weight = args.weight - 1
distort = args.distort
version = args.version

# figure out how to label the model name
if how == "label":
    model_label = "l"
elif how == "normal":
    model_label = "b"
elif how == "mask":
    model_label = "m"
else:
    model_label = "x"

# precalculated pixel mean of images
mu = 104.1353

# download the data
# download_data(what=dataset)

## config
batch_size = 16

train_files, total_records = get_training_data(what=dataset)

## Hyperparameters
epsilon = 1e-8

# learning rate
epochs_per_decay = 5
decay_factor = 0.75
staircase = True

# if we are retraining some layers start with smaller learning rate
if not stop and not freeze:
    starting_rate = 0.001
else:
    starting_rate = 0.0001

# learning rate decay variables
steps_per_epoch = int(total_records / batch_size)
print("Steps per epoch:", steps_per_epoch)

# lambdas
lamC = 0.00001
lamF = 0.00250

# use dropout
dropout = True
fcdropout_rate = 0.25
convdropout_rate = 0.001
pooldropout_rate = 0.1

if how == "label":
    num_classes = 5
elif how == "normal":
    num_classes = 2
elif how == "mass":
    num_classes = 3
elif how == "benign":
    num_classes = 3
elif how == "mask":
    num_classes = 2

print("Number of classes:", num_classes)

## Build the graph
graph = tf.Graph()

model_name = "model_4.0.0.01" + model_label + "." + str(dataset) + str(version)
## Change Log
# 4.0.0.01 - trying out a dense convnet

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
        with tf.device('/cpu:0'):
            image, label = read_and_decode_single_example(train_files, label_type=how, normalize=False, distort=False, size=640)

            X_def, y_def = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=2000, seed=None, min_after_dequeue=1000)

            # Placeholders
            X = tf.placeholder_with_default(X_def, shape=[None, 640, 640, 1])
            y = tf.placeholder_with_default(y_def, shape=[None, 640, 640, 1])

            X_fl = tf.cast(X, tf.float32)

            # optional online data augmentation
            if distort:
                X_dis, y_adj = augment(X_fl, y, horizontal_flip=True, augment_labels=True, vertical_flip=True, mixup=0)
            else:
                y_adj = y
                X_dis = X_fl

            # cast to float and scale input data
            X_adj = _scale_input_data(X_dis, contrast=contrast, mu=127.0, scale=255.0)

    # Convolutional layer 1 - output 320x320
    conv1 = _conv2d_batch_norm(X_adj, 32, kernel_size=(3,3), stride=(2,2), training=training, epsilon=1e-8, padding="SAME", seed=None, lambd=0.0, name="1.0", activation="relu")

    conv1 = _conv2d_batch_norm(conv1, 32, kernel_size=(3, 3), stride=(1, 1), training=training, epsilon=1e-8,
                               padding="SAME", seed=None, lambd=0.0, name="1.1", activation="relu")

    conv1 = _conv2d_batch_norm(conv1, 32, kernel_size=(3, 3), stride=(1, 1), training=training, epsilon=1e-8,
                               padding="SAME", seed=None, lambd=0.0, name="1.2", activation="relu")

    # Max pooling layer 1 - output 160x160
    with tf.name_scope('pool1') as scope:
        pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=(3, 3),
            strides=(2, 2),
            padding='SAME',
            name='pool1'
        )

        # optional dropout
        if dropout:
            pool1 = tf.layers.dropout(pool1, rate=pooldropout_rate, seed=103, training=training)

    ## Dense Layer 1
    dense1 = _dense_block(pool1, 6, growth_rate=12, bottleneck=True, training=training, seed=None, name="2.0", activation="relu")

    ## Transition 1 - ouput 80x80
    transition1 = _transition(dense1, filters=36, training=training, name="t_1")

    ## Dense Layer 2
    dense2 = _dense_block(transition1, 12, growth_rate=12, bottleneck=True, training=training, seed=None, name="3.0",
                          activation="relu")

    ## Transition 2 - output 40x40
    transition2 = _transition(dense2, filters=48, training=training, name="t_2")

    ## Dense Layer 3
    dense3 = _dense_block(transition2, 24, growth_rate=12, bottleneck=True, training=training, seed=None,
                          name="4.0",
                          activation="relu")

    ## Transition 3 - output 20x20
    transition3 = _transition(dense3, filters=56, training=training, name="t_3")

    ## Dense Layer 4
    dense4 = _dense_block(transition3, 16, growth_rate=12, bottleneck=True, training=training, seed=None,
                          name="5.0",
                          activation="relu")

    ## Transition 4 - output 10x10
    transition4 = _transition(dense4, filters=56, training=training, name="t_4")

    ## Dense 5
    dense5 = _dense_block(transition4, 16, growth_rate=12, bottleneck=True, training=training, seed=None,
                          name="6.0",
                          activation="relu")

    ## Transition 5 - output 5x5
    transition5 = _transition(dense5, filters=64, training=training, name="t_5")

    fc1 = _conv2d_batch_norm(transition5, 2048, kernel_size=(5, 5), stride=(5, 5), training=training, epsilon=1e-8,
                             padding="VALID", seed=1013, lambd=lamC, name="fc_1")

    fc1= tf.layers.dropout(fc1, rate=fcdropout_rate, seed=11537, training=training)

    fc2 = _conv2d_batch_norm(fc1, num_classes, kernel_size=(1, 1), stride=(1, 1), training=training, epsilon=1e-8,
                             padding="VALID", seed=1014, lambd=lamC, name="fc_2")

    fc2 = tf.layers.dropout(fc2, rate=fcdropout_rate, seed=12537, training=training)

    # upsample back to 5x5
    with tf.name_scope('up_conv1') as scope:
        unpool1 = tf.layers.conv2d_transpose(
            fc2,
            filters=64,
            kernel_size=(5, 5),
            strides=(5, 5),
            padding='SAME',
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=11435),
            kernel_regularizer=None,
            name='up_conv1'
        )

        unpool1 = unpool1 + transition5

    # upsample to 10x10
    with tf.name_scope('up_conv2') as scope:
        unpool2 = tf.layers.conv2d_transpose(
            unpool1,
            filters=56,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='SAME',
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=11435),
            kernel_regularizer=None,
            name='up_conv2'
        )

        # skip connection
        unpool2 = unpool2 + transition4

        unpool2 = tf.nn.elu(unpool2, name="up_conv2_relu")

        if dropout:
            unpool2 = tf.layers.dropout(unpool2, rate=convdropout_rate, seed=13537, training=training)

    # upsample to 20x20
    with tf.name_scope('up_conv3') as scope:
        unpool3 = tf.layers.conv2d_transpose(
            unpool2,
            filters=56,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='SAME',
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=19317),
            kernel_regularizer=None,
            name='up_conv3'
        )

        # skip connection
        unpool3 = unpool3 + transition3

        unpool3 = tf.nn.elu(unpool3, name='relu6.5')


    # upsample to 40x40
    with tf.name_scope('up_conv4') as scope:
        unpool4 = tf.layers.conv2d_transpose(
            unpool3,
            filters=48,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='SAME',
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=11728),
            kernel_regularizer=None,
            name='up_conv4'
        )

        if dropout:
            unpool4 = tf.layers.dropout(unpool4, rate=convdropout_rate, seed=14537, training=training)

        unpool4 = unpool4 + transition2

        unpool4 = tf.nn.elu(unpool4, name='up_relu4')

    # upsample to 80x80
    with tf.name_scope('up_conv5') as scope:
        unpool5 = tf.layers.conv2d_transpose(
            unpool4,
            filters=36,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='SAME',
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=11756),
            kernel_regularizer=None,
            name='up_conv5'
        )

        if dropout:
            unpool5 = tf.layers.dropout(unpool5, rate=pooldropout_rate, seed=14537, training=training)

        # skip connection
        unpool5 = unpool5 + transition1

        # activation
        unpool5 = tf.nn.elu(unpool5, name='relu10')

    conv6 = _conv2d_batch_norm(unpool5, 16, kernel_size=(3, 3), stride=(1, 1), training=training, lambd=0.0,
                               name="up_conv6", activation="elu")

    # upsample to 160x160
    with tf.name_scope('up_conv7') as scope:
        unpool7 = tf.layers.conv2d_transpose(
            conv6,
            filters=32,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='SAME',
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=11756),
            kernel_regularizer=None,
            name='up_conv7'
        )

        if dropout:
            unpool7 = tf.layers.dropout(unpool7, rate=pooldropout_rate, seed=14557, training=training)

            unpool7 = unpool7 + pool1

        # activation
        unpool7 = tf.nn.elu(unpool7, name='relu11')

    # one last conv layer before logits
    conv8 = _conv2d_batch_norm(unpool7, 16, kernel_size=(3,3), stride=(1,1), training=training, lambd=0.0, name="up_conv8", activation="elu")

    # upsample to 320x320
    with tf.name_scope('logits') as scope:
        logits = tf.layers.conv2d_transpose(
            conv8,
            filters=2,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='SAME',
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=11793),
            kernel_regularizer=None,
            name='logits'
        )

    # get the fully connected variables so we can only train them when retraining the network
    fc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "up_")
    tr_logits =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "logits")

    # with tf.variable_scope('conv1', reuse=True):
    #     conv_kernels1 = tf.get_variable('kernel')
    #     kernel_transposed = tf.transpose(conv_kernels1, [3, 0, 1, 2])
    #
    # with tf.variable_scope('visualization'):
    #     tf.summary.image('conv1/filters', kernel_transposed, max_outputs=32, collections=["kernels"])

    # This will weight the positive examples higher so as to improve recall
    weights = tf.multiply(tf.cast(weight, tf.float32), tf.cast(tf.greater(y_adj, 0), tf.float32)) + 1

    mean_ce = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y_adj, logits=(logits + 1e-10), weights=weights))

    # Add in l2 loss
    loss = mean_ce + tf.losses.get_regularization_loss()

    # Adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Minimize cross-entropy - freeze certain layers depending on input
    if freeze:
        train_op = optimizer.minimize(loss, global_step=global_step, var_list=fc_vars + tr_logits)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)

    # predictions = tf.reshape(tf.argmax(logits, axis=-1, output_type=tf.int32), (-1, 320,320))
    # if we reshape the predictions it won't work with images of other sizes
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

    # squash the predictions into a per image prediction - negative images will have a max of 0
    pred_sum = tf.reduce_sum(predictions, axis=[1, 2])
    image_predictions = tf.cast(tf.greater(pred_sum, (640 * 640 // 750)), dtype=tf.uint8)
    image_truth = tf.reduce_max(y_adj, axis=[1, 2])
    # image_predictions = tf.reduce_max(predictions, axis=[1,2,3])

    # set a threshold on the predictions so we ignore images with only a few positive pixels
    pred_sum = tf.reduce_sum(predictions, axis=[1, 2])
    image_predictions = tf.cast(tf.greater(pred_sum, (640*640//750)),dtype=tf.uint8)

    # get the accuracy per pixel
    # accuracy, acc_op = tf.metrics.accuracy(
    #     labels=y_adj,
    #     predictions=predictions,
    #     updates_collections=[tf.GraphKeys.UPDATE_OPS, 'metrics_ops'],
    #     name="accuracy",
    # )
    # # calculate recall and precision per pixel
    # recall, rec_op = tf.metrics.recall(labels=y_adj, predictions=predictions,
    #                                    updates_collections=[tf.GraphKeys.UPDATE_OPS, 'metrics_ops'],
    #                                    name="pixel_recall")
    # precision, prec_op = tf.metrics.precision(labels=y_adj, predictions=predictions,
    #                                           updates_collections=[tf.GraphKeys.UPDATE_OPS, 'metrics_ops'],
    #                                           name="pixel_precision")
    #
    # f1_score = 2 * ((precision * recall) / (precision + recall))
    #
    # # per image metrics
    # image_accuracy, image_acc_op = tf.metrics.accuracy(
    #     labels=image_truth,
    #     predictions=image_predictions,
    #     updates_collections=[tf.GraphKeys.UPDATE_OPS, 'metrics_ops'],
    #     name="image_accuracy",
    # )
    #
    # image_recall, image_rec_op = tf.metrics.recall(labels=image_truth, predictions=image_predictions,
    #                                    updates_collections=[tf.GraphKeys.UPDATE_OPS, 'metrics_ops'], name="image_recall")
    # image_precision, image_prec_op = tf.metrics.precision(labels=image_truth, predictions=image_predictions,
    #                                           updates_collections=[tf.GraphKeys.UPDATE_OPS, 'metrics_ops'], name="image_precision")
    #
    #
    # tf.summary.scalar('recall_1', recall, collections=["summaries"])
    # tf.summary.scalar('recall_per_image', image_recall, collections=["summaries"])
    # tf.summary.scalar('precision_1', precision, collections=["summaries"])
    # tf.summary.scalar('precision_per_image', image_precision, collections=["summaries"])
    # tf.summary.scalar('f1_score', f1_score, collections=["summaries"])
    #
    # # Create summary hooks
    # tf.summary.scalar('accuracy', accuracy, collections=["summaries"])
    # tf.summary.scalar('accuracy_per_image', image_accuracy, collections=["summaries"])
    # tf.summary.scalar('cross_entropy', mean_ce, collections=["summaries"])
    # tf.summary.scalar('learning_rate', learning_rate, collections=["summaries"])

    # add this so that the batch norm gets run
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # collect the metrics ops into one op so we can run that at test time
    metrics_op = tf.get_collection('metrics_ops')

    # Merge all the summaries
    merged = tf.summary.merge_all("summaries")
    kernel_summaries = tf.summary.merge_all("kernels")

    print("Graph created...")

## CONFIGURE OPTIONS
if init_model is not None:
    if os.path.exists(os.path.join("model", init_model + '.ckpt.index')):
        init = False
    else:
        init = True
elif restore_model is not None:
    if os.path.exists(os.path.join("model", restore_model + '.ckpt.index')):
        init = False
    else:
        init = True
else:
    if os.path.exists(os.path.join("model", model_name + '.ckpt.index')):
        init = False
    else:
        init = True

meta_data_every = 1
log_to_tensorboard = True
print_every = 5  # how often to print metrics
checkpoint_every = 1  # how often to save model in epochs
use_gpu = False  # whether or not to use the GPU
print_metrics = True  # whether to print or plot metrics, if False a plot will be created and updated every epoch

# Initialize metrics or load them from disk if they exist
if os.path.exists(os.path.join("data", model_name + "train_acc.npy")):
    train_acc_values = np.load(os.path.join("data", model_name + "train_acc.npy")).tolist()
else:
    train_acc_values = []

if os.path.exists(os.path.join("data", model_name + "train_loss.npy")):
    train_cost_values = np.load(os.path.join("data", model_name + "train_loss.npy")).tolist()
else:
    train_cost_values = []

if os.path.exists(os.path.join("data", model_name + "train_lr.npy")):
    train_lr_values = np.load(os.path.join("data", model_name + "train_lr.npy")).tolist()
else:
    train_lr_values = []

if os.path.exists(os.path.join("data", model_name + "train_recall.npy")):
    train_recall_values = np.load(os.path.join("data", model_name + "train_recall.npy")).tolist()
else:
    train_recall_values = []

if os.path.exists(os.path.join("data", model_name + "cv_acc.npy")):
    valid_acc_values = np.load(os.path.join("data", model_name + "cv_acc.npy")).tolist()
else:
    valid_acc_values = []

if os.path.exists(os.path.join("data", model_name + "cv_loss.npy")):
    valid_cost_values = np.load(os.path.join("data", model_name + "cv_loss.npy")).tolist()
else:
    valid_cost_values = []

if os.path.exists(os.path.join("data", model_name + "cv_recall.npy")):
    valid_recall_values = np.load(os.path.join("data", model_name + "cv_recall.npy")).tolist()
else:
    valid_recall_values = []

config = tf.ConfigProto()

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
        print("Initializing model...")
    else:
        # if we are initializing with the weights from another model load it
        if init_model is not None:
            # initialize the global variables
            sess.run(tf.global_variables_initializer())

            # create the initializer function to initialize the weights
            init_fn = load_weights(init_model, exclude=["fc3", "logits", "bn_conv6", "up_conv7",   "bn_up_conv8","bn_up_conv6","bn_up_conv7","conv_up_conv6", "conv_up_conv8","up_conv1","up_conv2","up_conv5","up_conv6", "accuracy", "up_conv4", "up_conv3", "global_step"])

            # run the initializer
            init_fn(sess)

            ## reload some weights from one checkpoint and some from a different one
            # init_fn = load_weights("model_s3.2.1.48m.12", exclude=["conv_up_conv7", "bn_up_conv7", "fc3", "conv5", "accuracy", "bn5"])
            # init_fn(sess)
            #
            # init_fn = load_weights("model_s3.2.0.47m.12", include=["conv5", "bn5"])
            # init_fn(sess)
            #
            # # reset the global step
            initial_global_step = global_step.assign(0)
            sess.run(initial_global_step)

            print("Initializing weights from model", init_model)

            # reset init model so we don't do this again
            init_model = None
        elif restore_model is not None:
            saver.restore(sess, './model/' + restore_model + '.ckpt')
            print("Restoring model", restore_model)

            initial_global_step = global_step.assign(0)
            sess.run(initial_global_step)
        # otherwise load this model
        else:
            saver.restore(sess, './model/' + model_name + '.ckpt')
            print("Restoring model", model_name)

    # if we are training the model
    if action == "train":

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print("Training model", model_name, "...")

        for epoch in range(epochs):
            sess.run(tf.local_variables_initializer())

            # Accuracy values (train) after each batch
            batch_acc = []
            batch_cost = []
            batch_recall = []

            for i in range(steps_per_epoch):
                # create the metadata
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                # Run training op and update ops
                if (i % 50 != 0) or (i == 0):
                    # log the kernel images once per epoch
                    if (i == (steps_per_epoch - 1)) and log_to_tensorboard:
                        _, _, image_summary, step = sess.run(
                            [train_op, extra_update_ops, kernel_summaries, global_step],
                            feed_dict={
                                training: True,
                            },
                            options=run_options,
                            run_metadata=run_metadata)

                        # write the summary
                        train_writer.add_summary(image_summary, step)
                    else:
                        _, _, step = sess.run(
                            [train_op, extra_update_ops, global_step],
                                feed_dict={
                                    training: True,
                                },
                                options=run_options,
                                run_metadata=run_metadata)

                # every 50th step get the metrics
                else:
                    _, _, precision_value, summary, acc_value, cost_value, recall_value, step, lr = sess.run(
                        [train_op, extra_update_ops, prec_op, merged, accuracy, mean_ce, rec_op, global_step, learning_rate],
                        feed_dict={
                            training: True,
                        },
                        options=run_options,
                        run_metadata=run_metadata)

                    # Save accuracy (current batch)
                    batch_acc.append(acc_value)
                    batch_cost.append(cost_value)
                    batch_recall.append(recall_value)

                    # log the summaries to tensorboard every 50 steps
                    if log_to_tensorboard:
                        # write the summary
                        train_writer.add_summary(summary, step)

                # only log the meta data once per epoch
                if i == 1:
                    train_writer.add_run_metadata(run_metadata, 'step %d' % step)

            # save checkpoint every nth epoch
            if (epoch % checkpoint_every == 0):
                print("Saving checkpoint")
                save_path = saver.save(sess, './model/' + model_name + '.ckpt')

                # Now that model is saved set init to false so we reload it next time
                init = False

            # init batch arrays
            batch_cv_acc = []
            batch_cv_loss = []
            batch_cv_recall = []

            # initialize the local variables so we have metrics only on the evaluation
            sess.run(tf.local_variables_initializer())

            print("Evaluating model...")
            # load the test data
            X_cv, y_cv = load_validation_data(percentage=1, how=how, which=dataset)

            # evaluate the test data
            for X_batch, y_batch in get_batches(X_cv, y_cv, batch_size, distort=False):
                _, valid_acc, valid_recall, valid_cost = sess.run(
                    [metrics_op, accuracy, recall, mean_ce],
                    feed_dict={
                        X: X_batch,
                        y: y_batch,
                        training: False
                    })

                batch_cv_acc.append(valid_acc)
                batch_cv_loss.append(valid_cost)
                batch_cv_recall.append(valid_recall)

            # Write average of validation data to summary logs
            if log_to_tensorboard:
                # evaluate once more to get the summary, which will then be written to tensorboard
                summary, cv_accuracy = sess.run(
                    [merged, accuracy],
                    feed_dict={
                        X: X_cv[0:2],
                        y: y_cv[0:2],
                        training: False
                    })

            test_writer.add_summary(summary, step)
            # test_writer.add_summary(other_summaries, step)
            step += 1

            # delete the test data to save memory
            del (X_cv)
            del (y_cv)

            print("Done evaluating...")

            # take the mean of the values to add to the metrics
            valid_acc_values.append(np.mean(batch_cv_acc))
            train_acc_values.append(np.mean(batch_acc))

            valid_cost_values.append(np.mean(batch_cv_loss))
            train_cost_values.append(np.mean(batch_cost))

            valid_recall_values.append(np.mean(batch_cv_recall))
            train_recall_values.append(np.mean(batch_recall))

            train_lr_values.append(lr)

            # save the metrics
            np.save(os.path.join("data", model_name + "train_acc.npy"), train_acc_values)
            np.save(os.path.join("data", model_name + "cv_acc.npy"), valid_acc_values)

            np.save(os.path.join("data", model_name + "train_loss.npy"), train_cost_values)
            np.save(os.path.join("data", model_name + "cv_loss.npy"), valid_cost_values)

            np.save(os.path.join("data", model_name + "train_recall.npy"), train_recall_values)
            np.save(os.path.join("data", model_name + "cv_recall.npy"), valid_recall_values)

            np.save(os.path.join("data", model_name + "train_lr.npy"), train_lr_values)

            # Print progress every nth epoch to keep output to reasonable amount
            if (epoch % print_every == 0):
                print(
                'Epoch {:02d} - step {} - cv acc: {:.4f} - train acc: {:.3f} (mean)'.format(
                    epoch, step, np.mean(batch_cv_acc), np.mean(batch_acc)
                ))

        # stop the coordinator
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)

    sess.run(tf.local_variables_initializer())
    print("Evaluating on test data")

    # evaluate the test data
    X_te, y_te = load_validation_data(how=how, data="test", which=dataset)

    test_accuracy = []
    test_recall = []
    test_predictions = []
    ground_truth = []
    for X_batch, y_batch in get_batches(X_te, y_te, batch_size, distort=False):
        yhat, test_acc_value, test_recall_value, test_prec_value = sess.run([predictions, acc_op, rec_op, prec_op], feed_dict=
        {
            X: X_batch,
            y: y_batch,
            training: False
        })

        test_accuracy.append(test_acc_value)
        test_recall.append(test_recall_value)
        test_predictions.append(yhat)
        ground_truth.append(y_batch)

    # print the results
    print("Mean Test Accuracy:", np.mean(test_accuracy))
    print("Mean Test Recall:", np.mean(test_recall))

    # unlist the predictions and truth
    test_predictions = flatten(test_predictions)
    ground_truth = flatten(ground_truth)

    # save the predictions and truth for review
    # np.save(os.path.join("data", "predictions_" + model_name + ".npy"), test_predictions)
    # np.save(os.path.join("data", "truth_" + model_name + ".npy"), ground_truth)

    sess.run(tf.local_variables_initializer())


