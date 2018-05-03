import numpy as np
import os
import wget
from sklearn.model_selection import train_test_split
import tensorflow as tf
from training_utils import download_file, get_batches, read_and_decode_single_example, load_validation_data, \
    download_data, evaluate_model, get_training_data, load_weights, flatten, _conv2d_batch_norm
import argparse
from tensorboard import summary as summary_lib

# If number of epochs has been passed in use that, otherwise default to 50
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", help="number of epochs to train", default=30, type=int)
parser.add_argument("-d", "--data", help="which dataset to use", default=9, type=int)
parser.add_argument("-m", "--model", help="model to initialize with", default=None)
parser.add_argument("-l", "--label", help="how to classify data", default="normal")
parser.add_argument("-a", "--action", help="action to perform", default="train")
parser.add_argument("-t", "--threshold", help="decision threshold", default=0.4, type=float)
args = parser.parse_args()

epochs = args.epochs
dataset = args.data
init_model = args.model
how = args.label
action = args.action
threshold = args.threshold

# precalculated pixel mean of images
mu = 104.1353

# download the data
download_data(what=dataset)

batch_size = 32

train_files, total_records = get_training_data(what=dataset)

## Hyperparameters
# Small epsilon value for the BN transform
epsilon = 1e-8

# learning rate
epochs_per_decay = 10
starting_rate = 0.001
decay_factor = 0.80
staircase = True

# learning rate decay variables
steps_per_epoch = int(total_records / batch_size)
print("Steps per epoch:", steps_per_epoch)

# lambdas
lamC = 0.00000
lamF = 0.00250

# use dropout
dropout = False
fcdropout_rate = 0.5
convdropout_rate = 0.0
pooldropout_rate = 0.0

if how == "label":
    num_classes = 5
elif how == "normal":
    num_classes = 2
elif how == "mass":
    num_classes = 3
elif how == "benign":
    num_classes = 3

print("Number of classes:", num_classes)

## Build the graph
graph = tf.Graph()

# whether to retrain model from scratch or use saved model
init = True
model_name = "vgg_16.2.04b.6"
# vgg_19.01 - attempting to recreate vgg 19 architecture
# vgg_16.02 - went to vgg 16 architecture, reducing units in fc layers
# vgg_16.2.01 - changing first conv layers to stride 2 to get dimensions down to reasonable size
# vgg_16.2.02 - using normal x-entropy instead of weighted

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
        image, label = read_and_decode_single_example(train_files, label_type=how, normalize=False)

        X_def, y_def = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=2000,
                                              min_after_dequeue=1000)

        # Placeholders
        X = tf.placeholder_with_default(X_def, shape=[None, 299, 299, 1])
        y = tf.placeholder_with_default(y_def, shape=[None])

        X = tf.cast(X, dtype=tf.float32)

        # center the pixel data
        mu = tf.constant(mu, name="pixel_mean", dtype=tf.float32)
        X = tf.subtract(X, mu, name="centered_input")

    # Convolutional layer 1
    conv1 = _conv2d_batch_norm(X, 64, kernel_size=(3,3), stride=(2,2), training=training, epsilon=1e-8, padding="SAME", seed=100, lambd=lamC, name="1.1")
    conv1 = _conv2d_batch_norm(conv1, 64, kernel_size=(3, 3), stride=(1, 1), training=training, epsilon=1e-8, padding="SAME", seed=100, lambd=lamC, name="1.2")

    # Max pooling layer 1
    with tf.name_scope('pool1') as scope:
        pool1 = tf.layers.max_pooling2d(
            conv1,  # Input
            pool_size=(2, 2),  # Pool size: 2x2
            strides=(2, 2),  # Stride: 2
            padding='SAME',  # "same" padding
            name='pool1'
        )

        # optional dropout
        if dropout:
            pool1 = tf.layers.dropout(pool1, rate=pooldropout_rate, seed=103, training=training)

    # Layer 2
    conv2 = _conv2d_batch_norm(pool1, 128, kernel_size=(3, 3), stride=(1, 1), training=training, epsilon=1e-8, padding="SAME", seed=None, lambd=lamC, name="2.1")

    conv2 = _conv2d_batch_norm(conv2, 128, kernel_size=(3, 3), stride=(1, 1), training=training, epsilon=1e-8, padding="SAME", seed=None, lambd=lamC, name="2.2")

    # Max pooling layer 1
    with tf.name_scope('pool2') as scope:
        pool2 = tf.layers.max_pooling2d(
            conv2,  # Input
            pool_size=(2, 2),  # Pool size: 3x3
            strides=(2, 2),  # Stride: 2
            padding='SAME',  # "same" padding
            name='pool2'
        )

        # optional dropout
        if dropout:
            pool2 = tf.layers.dropout(pool2, rate=pooldropout_rate, seed=103, training=training)

    # Convolutional layer 3
    conv3 = _conv2d_batch_norm(pool2, 256, kernel_size=(3, 3), stride=(1, 1), training=training, epsilon=1e-8, padding="SAME", seed=None, lambd=lamC, name="3.1")

    conv3 = _conv2d_batch_norm(conv3, 256, kernel_size=(3, 3), stride=(1, 1), training=training, epsilon=1e-8, padding="SAME", seed=None, lambd=lamC, name="3.2")

    # Max pooling layer 3
    with tf.name_scope('pool3') as scope:
        pool3 = tf.layers.max_pooling2d(
            conv3,
            pool_size=(2, 2),  # Pool size: 2x2
            strides=(2, 2),  # Stride: 2
            padding='SAME',
            name='pool5'
        )

        if dropout:
            pool3 = tf.layers.dropout(pool3, rate=pooldropout_rate, seed=115, training=training)

    conv4 = _conv2d_batch_norm(pool3, 512, kernel_size=(3, 3), stride=(1, 1), training=training, epsilon=1e-8, padding="SAME", seed=None, lambd=lamC, name="4.1")

    conv4 = _conv2d_batch_norm(conv4, 512, kernel_size=(3, 3), stride=(1, 1), training=training, epsilon=1e-8, padding="SAME", seed=None, lambd=lamC, name="4.2")

    conv4 = _conv2d_batch_norm(conv4, 512, kernel_size=(3, 3), stride=(1, 1), training=training, epsilon=1e-8, padding="SAME", seed=None, lambd=lamC, name="4.3")

    # Max pooling layer 4
    with tf.name_scope('pool4') as scope:
        pool4 = tf.layers.max_pooling2d(
            conv4,
            pool_size=(2, 2),  # Pool size: 2x2
            strides=(2, 2),  # Stride: 2
            padding='SAME',
            name='pool4'
        )

        if dropout:
            pool4 = tf.layers.dropout(pool4, rate=pooldropout_rate, seed=115, training=training)

    conv5 = _conv2d_batch_norm(pool4, 512, kernel_size=(3, 3), stride=(1, 1), training=training, epsilon=1e-8, padding="SAME", seed=None, lambd=lamC, name="5.1")

    conv5 = _conv2d_batch_norm(conv5, 512, kernel_size=(3, 3), stride=(1, 1), training=training, epsilon=1e-8, padding="SAME", seed=None, lambd=lamC, name="5.2")

    conv5 = _conv2d_batch_norm(conv5, 512, kernel_size=(3, 3), stride=(1, 1), training=training, epsilon=1e-8, padding="SAME", seed=None, lambd=lamC, name="5.3")

    # Max pooling layer 5
    with tf.name_scope('pool5') as scope:
        pool5 = tf.layers.max_pooling2d(
            conv5,
            pool_size=(2, 2),  # Pool size: 2x2
            strides=(2, 2),  # Stride: 2
            padding='SAME',
            name='pool5'
        )

        if dropout:
            pool5 = tf.layers.dropout(pool5, rate=pooldropout_rate, seed=115, training=training)

    # Flatten output
    with tf.name_scope('flatten') as scope:
        flat_output = tf.contrib.layers.flatten(pool5)

        # dropout at fc rate
        flat_output = tf.layers.dropout(flat_output, rate=fcdropout_rate, seed=116, training=training)

    # Fully connected layer 1
    with tf.name_scope('fc1') as scope:
        fc1 = tf.layers.dense(
            flat_output,
            2048,
            activation=None,
            kernel_initializer=tf.variance_scaling_initializer(scale=2, seed=117),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamF),
            name="fc1"
        )

        fc1 = tf.layers.batch_normalization(
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
            name='fc1_bn'
        )

        fc1 = tf.nn.relu(fc1, name='fc1_relu')

        # dropout
        fc1 = tf.layers.dropout(fc1, rate=fcdropout_rate, seed=118, training=training)

    # Fully connected layer 2
    with tf.name_scope('fc2') as scope:
        fc2 = tf.layers.dense(
            fc1,  # input
            2048,  # 1024 hidden units
            activation=None,  # None
            kernel_initializer=tf.variance_scaling_initializer(scale=2, seed=119),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamF),
            name="fc2"
        )

        fc2 = tf.layers.batch_normalization(
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
            name='fc2_bn'
        )

        fc2 = tf.nn.relu(fc2, name='fc2_relu')

        # dropout
        fc2 = tf.layers.dropout(fc2, rate=fcdropout_rate, seed=120, training=training)

    # Output layer
    logits = tf.layers.dense(
        fc2,
        num_classes,  # One output unit per category
        activation=None,  # No activation function
        kernel_initializer=tf.variance_scaling_initializer(scale=1, seed=121),
        bias_initializer=tf.zeros_initializer(),
        name="fc_logits"
    )

    # get the fully connected variables so we can only train them when retraining the network
    fc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc")

    with tf.variable_scope('conv_1.1', reuse=True):
        conv_kernels1 = tf.get_variable('kernel')
        kernel_transposed = tf.transpose(conv_kernels1, [3, 0, 1, 2])

    with tf.variable_scope('visualization'):
        tf.summary.image('conv_1.1/filters', kernel_transposed, max_outputs=32, collections=["kernels"])

    # get the probabilites for the classes
    probabilities = tf.nn.softmax(logits, name="probabilities")

    # the probability that the scan is abnormal is 1 - probability it is normal
    abnormal_probability = (1 - probabilities[:, 0])

    if num_classes > 2:
        # the scan is abnormal if the probability is greater than the threshold
        is_abnormal = tf.cast(tf.greater(abnormal_probability, threshold), tf.int64)

        # Compute predictions from the probabilities - if the scan is normal we ignore the other probabilities
        predictions = is_abnormal * tf.argmax(probabilities[:,1:], axis=1, output_type=tf.int64)
    else:
        predictions = tf.cast(tf.greater(abnormal_probability, threshold), tf.int32)

    # get the accuracy
    accuracy, acc_op = tf.metrics.accuracy(
        labels=y,
        predictions=predictions,
        updates_collections=tf.GraphKeys.UPDATE_OPS,
        # metrics_collections=["summaries"],
        name="accuracy",
    )

    #########################################################
    ## Loss function options
    # Regular mean cross entropy
    mean_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

    #########################################################
    ## Weight the positive examples higher
    # This will weight the positive examples higher so as to improve recall
    #weights = tf.multiply(1, tf.cast(tf.greater(y, 0), tf.int32)) + 1
    #mean_ce = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits, weights=weights))

    # Add in l2 loss
    loss = mean_ce + tf.losses.get_regularization_loss()

    # Adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Minimize cross-entropy
    train_op = optimizer.minimize(loss, global_step=global_step)

    # calculate recall
    if num_classes > 2:
        # collapse the predictions down to normal or not for our pr metrics
        zero = tf.constant(0, dtype=tf.int64)
        collapsed_predictions = tf.cast(tf.greater(predictions, zero), tf.int64)
        collapsed_labels = tf.greater(y, zero)

        recall, rec_op = tf.metrics.recall(labels=collapsed_labels, predictions=collapsed_predictions, updates_collections=tf.GraphKeys.UPDATE_OPS, name="recall")
        precision, prec_op = tf.metrics.precision(labels=collapsed_labels, predictions=collapsed_predictions, updates_collections=tf.GraphKeys.UPDATE_OPS, name="precision")

    else:
        recall, rec_op = tf.metrics.recall(labels=y, predictions=predictions, updates_collections=tf.GraphKeys.UPDATE_OPS, name="recall")
        precision, prec_op = tf.metrics.precision(labels=y, predictions=predictions, updates_collections=tf.GraphKeys.UPDATE_OPS, name="precision")

    f1_score = 2 * ((precision * recall) / (precision + recall))
    _, update_op = summary_lib.pr_curve_streaming_op(name='pr_curve',
                                                     predictions=(1 - probabilities[:, 0]),
                                                     labels=y,
                                                     updates_collections=tf.GraphKeys.UPDATE_OPS,
                                                     num_thresholds=20)

    tf.summary.scalar('recall_1', recall, collections=["summaries"])
    tf.summary.scalar('precision_1', precision, collections=["summaries"])
    tf.summary.scalar('f1_score', f1_score, collections=["summaries"])

    # Create summary hooks
    tf.summary.scalar('accuracy', accuracy, collections=["summaries"])
    tf.summary.scalar('cross_entropy', mean_ce, collections=["summaries"])
    tf.summary.scalar('learning_rate', learning_rate, collections=["summaries"])

    # add this so that the batch norm gets run
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Merge all the summaries
    merged = tf.summary.merge_all("summaries")
    kernel_summaries = tf.summary.merge_all("kernels")
    per_epoch_summaries = [[]]

    print("Graph created...")

## CONFIGURE OPTIONS
if init_model is not None:
    if os.path.exists(os.path.join("model", init_model + '.ckpt.index')):
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
        f, ax = plt.subplots(1, 4, figsize=(24, 5))

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
            init_fn = load_weights(init_model, exclude=["fc1", "fc1_bn", "fc2_bn", "fc2", "fc_logits", "global_step"])

            # run the initializer
            init_fn(sess)

            # reset the global step
            initial_global_step = global_step.assign(0)
            sess.run(initial_global_step)

            print("Initializing weights from model", init_model)

            # reset init model so we don't do this again
            init_model = None
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
                        _, _, _, image_summary, step = sess.run(
                            [train_op, extra_update_ops, update_op, kernel_summaries, global_step],
                            feed_dict={
                                training: True,
                            },
                            options=run_options,
                            run_metadata=run_metadata)

                        # write the summary
                        train_writer.add_summary(image_summary, step)
                    else:
                        _, _, _, step = sess.run(
                            [train_op, extra_update_ops, update_op, global_step],
                                feed_dict={
                                    training: True,
                                },
                                options=run_options,
                                run_metadata=run_metadata)

                # every 50th step get the metrics
                else:
                    _, _, _, precision_value, summary, acc_value, cost_value, recall_value, step, lr = sess.run(
                        [train_op, extra_update_ops, update_op, prec_op, merged, accuracy, mean_ce, rec_op, global_step, learning_rate],
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
                _, _, valid_acc, valid_recall, valid_precision, valid_fscore, valid_cost = sess.run(
                    [update_op, extra_update_ops, accuracy, rec_op, prec_op, f1_score, mean_ce],
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

            # Print data every 50th epoch so I can write it down to compare models
            if (not print_metrics) and (epoch % 50 == 0) and (epoch > 1):
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
        _, yhat, test_acc_value, test_recall_value = sess.run([extra_update_ops, predictions, accuracy, rec_op], feed_dict=
        {
            X: X_batch,
            y: y_batch,
            training: False
        })

        test_accuracy.append(test_acc_value)
        test_recall.append(test_recall_value)
        test_predictions.append(yhat)
        ground_truth.append(y_batch)

    print("Evaluating on test data")

    # print the results
    print("Mean Test Accuracy:", np.mean(test_accuracy))
    print("Mean Test Recall:", np.mean(test_recall))

    # unlist the predictions and truth
    test_predictions = flatten(test_predictions)
    ground_truth = flatten(ground_truth)

    # save the predictions and truth for review
    np.save(os.path.join("data", "predictions_" + model_name + ".npy"), test_predictions)
    np.save(os.path.join("data", "truth_" + model_name + ".npy"), ground_truth)

    sess.run(tf.local_variables_initializer())

    ## evaluate on MIAS  data
    X_te, y_te = load_validation_data(how=how, data="mias", which=dataset)

    mias_test_accuracy = []
    mias_test_recall = []
    mias_test_predictions = []
    mias_ground_truth = []
    for X_batch, y_batch in get_batches(X_te, y_te, batch_size, distort=False):
        _, yhat, test_acc_value, test_recall_value = sess.run([extra_update_ops, predictions, accuracy, rec_op], feed_dict=
        {
            X: X_batch,
            y: y_batch,
            training: False
        })

        mias_test_accuracy.append(test_acc_value)
        mias_test_recall.append(test_recall_value)
        mias_test_predictions.append(yhat)
        mias_ground_truth.append(y_batch)

    # print the results
    print("Mean MIAS Accuracy:", np.mean(mias_test_accuracy))
    print("Mean MIAS Recall:", np.mean(mias_test_recall))

    # unlist the predictions and truth
    mias_test_predictions = flatten(mias_test_predictions)
    mias_ground_truth = flatten(mias_ground_truth)

    # save the predictions and truth for review
    np.save(os.path.join("data", "mias_predictions_" + model_name + ".npy"), mias_test_predictions)
    np.save(os.path.join("data", "mias_truth_" + model_name + ".npy"), mias_ground_truth)


