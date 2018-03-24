import tarfile
import numpy as np
import re
import pickle
import os
import wget
from sklearn.cross_validation  import train_test_split
import tensorflow as tf

## download a file to a location in the data folder
def download_file(url, name):
    print("Downloading " + name + "...")
    
    # check that the data directory exists
    try:
        os.stat("data")
    except:
        os.mkdir("data")  
    fname = wget.download(url, os.path.join('data',name)) 
    
## Batch generator
def get_batches(X, y, batch_size, distort=True):
    # Shuffle X,y
    shuffled_idx = np.arange(len(y))
    np.random.shuffle(shuffled_idx)
    i, h, w, c = X.shape
    
    # Enumerate indexes by steps of batch_size
    for i in range(0, len(y), batch_size):
        batch_idx = shuffled_idx[i:i+batch_size]
        X_return = X[batch_idx]
        
        # do random flipping of images
        coin = np.random.binomial(1, 0.5, size=None)
        if coin and distort:
            X_return = X_return[...,::-1,:]
        
        yield X_return, y[batch_idx]

## read data from tfrecords file        
def read_and_decode_single_example(filenames, label_type='label', num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
    
    reader = tf.TFRecordReader()
    
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'label_normal': tf.FixedLenFeature([], tf.int64),
            'label_mass': tf.FixedLenFeature([], tf.int64),
            'label_benign': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)
        })
    
    # extract the data
    label = features[label_type]
    image = tf.decode_raw(features['image'], tf.uint8)
    
    # reshape and scale the image
    image = tf.reshape(image, [299, 299, 1])
    image = tf.image.per_image_standardization(image)
    
    # return the image and the label
    return image, label

## load the test data from files
def load_validation_data(how="class", percentage=0.5):
    X_cv = np.load(os.path.join("data", "test_data.npy"))
    labels = np.load(os.path.join("data", "test_labels.npy"))
    
    # encode the labels appropriately
    if how == "class":
        y_cv = labels
    elif how == "normal":
        y_cv = np.zeros(len(labels))
        y_cv[labels != 0] = 1
    elif how == "mass":
        y_cv = np.zeros(len(labels))
        y_cv[labels == 1] = 1
        y_cv[labels == 3] = 1
        y_cv[labels == 2] = 2
        y_cv[labels == 4] = 4
    elif how == "benign":
        y_cv = np.zeros(len(labels))
        y_cv[labels == 1] = 1
        y_cv[labels == 2] = 1
        y_cv[labels == 3] = 2
        y_cv[labels == 4] = 2

    # shuffle the data
    X_cv, _, y_cv, _ = train_test_split(X_cv, y_cv, test_size=1-percentage)
    
    return X_cv, y_cv

def evaluate(graph, config):
    X_cv, y_cv = load_validation_data(how="normal")

    with tf.Session(graph=graph, config=config) as sess:
        # create the saver
        saver = tf.train.Saver()
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, './model/' + model_name + '.ckpt')

        yhat, test_acc, test_recall = sess.run([predictions, accuracy, rec_op], feed_dict=
        {
            X: X_cv[0:128],
            y: y_cv[0:128],
            training: False
        })

    # initialize the counter
    i = 0

    # print the results
    print("Recall:", test_recall)
    print("Accuracy:", test_acc)

    # print specifics
    for item in yhat:
        if item == y_cv[i]:
            found = "*"
        else:
            found = ""

        print("Prediction:", item, " Actual:", y_cv[i], found)
        i += 1

## Download the data if it doesn't already exist
def download_data():
    if not os.path.exists(os.path.join("data","training_0.tfrecords")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training_0.tfrecords', 'training_0.tfrecords')

    if not os.path.exists(os.path.join("data","training_1.tfrecords")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training_1.tfrecords', 'training_1.tfrecords')

    if not os.path.exists(os.path.join("data","training_2.tfrecords")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training_2.tfrecords', 'training_2.tfrecords')

    if not os.path.exists(os.path.join("data","training_3.tfrecords")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training_3.tfrecords', 'training_3.tfrecords')

    if not os.path.exists(os.path.join("data","test_labels.npy")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test_labels.npy', 'test_labels.npy')

    if not os.path.exists(os.path.join("data","test_data.npy")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test_data.npy', 'test_data.npy')


# ## Create Model

# config
epochs = 100                  
batch_size = 32

## Hyperparameters
# Small epsilon value for the BN transform
epsilon = 1e-8

# learning rate
epochs_per_decay = 25
starting_rate = 0.002
decay_factor = 0.85
staircase = True

# learning rate decay variables
steps_per_epoch = int(22000 / batch_size)
print("Steps per epoch:", steps_per_epoch)

# lambdas
lamC = 0.00010
lamF = 0.00100

# use dropout
dropout = True

num_classes = 2

train_path_0 = os.path.join("data", "training_0.tfrecords")
train_path_1 = os.path.join("data", "training_1.tfrecords")
train_path_2 = os.path.join("data", "training_2.tfrecords")
train_path_3 = os.path.join("data", "training_3.tfrecords")
test_path = os.path.join("data", "test.tfrecords")
train_files = [train_path_0, train_path_1, train_path_2, train_path_3]

# train the model
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
    else:
        saver.restore(sess, './model/' + model_name + '.ckpt')

    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print("Training model", model_name, "...")

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
            _, _, summary, acc_value, cost_value, loss_value, recall_value, step, lr = sess.run(
                [train_op, extra_update_ops,
                 merged, accuracy, mean_ce, loss, rec_op, global_step,
                 learning_rate], feed_dict={
                    # X: X_batch,
                    # y: y_batch,
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

        # save checkpoint every nth epoch
        if (epoch % checkpoint_every == 0):
            print("Saving checkpoint")
            # save_path = saver.save(sess, './model/'+model_name+'.ckpt')

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
            X_cv, y_cv = load_validation_data(percentage=0.2, how="normal")

            # evaluate the test data
            for X_batch, y_batch in get_batches(X_cv, y_cv, batch_size // 2, distort=False):
                summary, valid_acc, valid_recall, valid_cost, valid_loss = sess.run(
                    [merged, accuracy, rec_op, mean_ce, loss],
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
                                            tf.Summary.Value(tag="cross_entropy",
                                                             simple_value=np.mean(batch_cv_cost)), ])
                test_writer.add_summary(summary, step)
                step += 1

            # delete the test data to save memory
            del (X_cv)
            del (y_cv)

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

        if print_metrics:
            # Print progress every nth epoch to keep output to reasonable amount
            if (epoch % print_every == 0):
                print(
                'Epoch {:02d} - step {} - cv acc: {:.3f} - train acc: {:.3f} (mean) - cv cost: {:.3f} - lr: {:.5f}'.format(
                    epoch, step, np.mean(batch_cv_acc), np.mean(batch_acc), np.mean(batch_cv_cost), lr
                ))
        else:
            # update the plot
            ax[0].cla()
            ax[0].plot(valid_acc_values, color="red", label="Validation")
            ax[0].plot(train_acc_values, color="blue", label="Training")
            ax[0].set_title('Validation accuracy: {:.4f} (mean last 3)'.format(np.mean(valid_acc_values[-4:])))

            # since we can't zoom in on plots like in tensorboard, scale y axis to give a decent amount of detail
            # if np.mean(valid_acc_values[-3:]) > 0.90:
            #    ax[0].set_ylim([0.80,1.0])
            # elif np.mean(valid_acc_values[-3:]) > 0.85:
            #    ax[0].set_ylim([0.75,1.0])
            # elif np.mean(valid_acc_values[-3:]) > 0.75:
            #    ax[0].set_ylim([0.65,1.0])
            # elif np.mean(valid_acc_values[-3:]) > 0.65:
            #    ax[0].set_ylim([0.55,1.0])
            # elif np.mean(valid_acc_values[-3:]) > 0.55:
            #    ax[0].set_ylim([0.45,1.0])

            ax[0].set_xlabel('Epoch')
            ax[0].set_ylabel('Accuracy')
            ax[0].legend()

            ax[1].cla()
            ax[1].plot(valid_recall_values, color="red", label="Validation")
            ax[1].plot(train_recall_values, color="blue", label="Training")
            ax[1].set_title('Validation recall: {:.3f} (mean last 3)'.format(np.mean(valid_recall_values[-4:])))
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('Recall')
            # ax[1].set_ylim([0,2.0])
            ax[1].legend()

            ax[2].cla()
            ax[2].plot(valid_cost_values, color="red", label="Validation")
            ax[2].plot(train_cost_values, color="blue", label="Training")
            ax[2].set_title('Validation xentropy: {:.3f} (mean last 3)'.format(np.mean(valid_cost_values[-4:])))
            ax[2].set_xlabel('Epoch')
            ax[2].set_ylabel('Cross Entropy')
            ax[2].set_ylim([0, 3.0])
            ax[2].legend()

            ax[3].cla()
            ax[3].plot(train_lr_values)
            ax[3].set_title("Learning rate: {:.6f}".format(np.mean(train_lr_values[-1:])))
            ax[3].set_xlabel("Epoch")
            ax[3].set_ylabel("Learning Rate")

            display.display(plt.gcf())
            display.clear_output(wait=True)

        # Print data every 50th epoch so I can write it down to compare models
        if (not print_metrics) and (epoch % 50 == 0) and (epoch > 1):
            if (epoch % print_every == 0):
                print(
                'Epoch {:02d} - step {} - cv acc: {:.4f} - train acc: {:.3f} (mean) - cv cost: {:.3f} - lr: {:.5f}'.format(
                    epoch, step, np.mean(batch_cv_acc), np.mean(batch_acc), np.mean(batch_cv_cost), lr
                ))

                # stop the coordinator
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)

# ## Train

## CONFIGURE OPTIONS
init = True                   # whether to initialize the model or use a saved version
crop = False                  # do random cropping of images?

meta_data_every = 5
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
            X_cv, y_cv = load_validation_data(percentage=0.5, how="normal")
            
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
                                            tf.Summary.Value(tag="cross_entropy", simple_value=np.mean(batch_cv_cost)),])
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

