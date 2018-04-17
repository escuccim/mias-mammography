import numpy as np
import os
import wget
import zipfile
from sklearn.model_selection import train_test_split
import tensorflow as tf

## open zip files
def unzip(file, destination):
    foo = zipfile.ZipFile(file, mode='r', allowZip64=True)
    foo.extractall(destination)

    return True

## download a file to a location in the data folder. If the file is a zip file unzip it and delete
## the archive to save disk space
def download_file(url, name):
    print("\nDownloading " + name + "...")

    # check that the data directory exists
    try:
        os.stat("data")
    except:
        os.mkdir("data")

    try:
        fname = wget.download(url, os.path.join('data', name))

        # if the file is a zip file unzip it
        if "zip" in name:
            unzip(os.path.join("data", name), "data")

            # then delete the zip to save disk space
            try:
                os.remove(os.path.join("data", name))
                print("Zip file extracted and deleted", name)
            except:
                print("Error deleting zip file", name)

    except:
        print("Error downloading", url)



## Batch generator with optional filenames parameter which will also return the filenames of the images
## so that they can be identified
def get_batches(X, y, batch_size, filenames=None, distort=False):
    # Shuffle X,y
    shuffled_idx = np.arange(len(y))
    np.random.shuffle(shuffled_idx)
    i, h, w, c = X.shape

    # Enumerate indexes by steps of batch_size
    for i in range(0, len(y), batch_size):
        batch_idx = shuffled_idx[i:i + batch_size]
        X_return = X[batch_idx]

        # do random flipping of images
        coin = np.random.binomial(1, 0.5, size=None)
        if coin and distort:
            X_return = X_return[..., ::-1, :]

        if filenames is None:
            yield X_return, y[batch_idx]
        else:
            yield X_return, y[batch_idx], filenames[batch_idx]

## read data from tfrecords file
def read_and_decode_single_example(filenames, label_type='label_normal', normalize=False, distort=False, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)

    reader = tf.TFRecordReader()

    if label_type != 'label':
        label_type = 'label_' + label_type

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

    if normalize:
        image = tf.image.per_image_standardization(image)

    # random flipping of image
    if distort:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

    # return the image and the label
    return image, label


## load the test data from files
def load_validation_data(data="validation", how="normal", which=5, percentage=1):
    if data == "validation":
        # load the two data files
        if which == 4:
            X_cv = np.load(os.path.join("data", "cv4_data.npy"))
            labels = np.load(os.path.join("data", "cv4_labels.npy"))
        elif which == 5:
            X_cv = np.load(os.path.join("data", "cv5_data.npy"))
            labels = np.load(os.path.join("data", "cv5_labels.npy"))
        elif which == 6:
            X_cv = np.load(os.path.join("data", "cv6_data.npy"))
            labels = np.load(os.path.join("data", "cv6_labels.npy"))

    elif data == "test":
        if which == 4:
            X_cv = np.load(os.path.join("data", "test4_data.npy"))
            labels = np.load(os.path.join("data", "test4_labels.npy"))
        elif which == 5:
            X_cv = np.load(os.path.join("data", "test5_data.npy"))
            labels = np.load(os.path.join("data", "test5_labels.npy"))
        elif which == 6:
            X_cv = np.load(os.path.join("data", "test6_data.npy"))
            labels = np.load(os.path.join("data", "test6_labels.npy"))

    elif data == "mias":
        X_cv = np.load(os.path.join("data", "mias_test_images.npy"))
        labels = np.load(os.path.join("data", "mias_test_labels_enc.npy"))

    # encode the labels appropriately
    if how == "label":
        y_cv = labels
    elif how == "normal":
        y_cv = np.zeros(len(labels))
        y_cv[labels != 0] = 1
    elif how == "mass":
        y_cv = np.zeros(len(labels))
        y_cv[labels == 1] = 1
        y_cv[labels == 3] = 1
        y_cv[labels == 2] = 2
        y_cv[labels == 4] = 2
    elif how == "benign":
        y_cv = np.zeros(len(labels))
        y_cv[labels == 1] = 1
        y_cv[labels == 2] = 1
        y_cv[labels == 3] = 2
        y_cv[labels == 4] = 2

    # shuffle the data
    X_cv, _, y_cv, _ = train_test_split(X_cv, y_cv, test_size=1 - percentage)

    return X_cv, y_cv

## Download the data if it doesn't already exist, many datasets have been created, which one to download can be specified using
## the what argument
def download_data(what=4):
    if what == 4:
        # download and unzip tfrecords training data
        if not os.path.exists(os.path.join("data", "training4_0.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training4_0.zip',
                              'training4_0.zip')

        if not os.path.exists(os.path.join("data", "training4_1.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training4_1.zip',
                              'training4_1.zip')

        if not os.path.exists(os.path.join("data", "training4_2.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training4_2.zip',
                              'training4_2.zip')

        if not os.path.exists(os.path.join("data", "training4_3.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training4_3.zip',
                              'training4_3.zip')

        if not os.path.exists(os.path.join("data", "training4_4.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training4_4.zip',
                              'training4_4.zip')

        # download and unzip test data
        if not os.path.exists(os.path.join("data", "test4_data.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test4_data.zip',
                              'test4_data.zip')

        if not os.path.exists(os.path.join("data", "test4_filenames.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test4_filenames.npy',
                              'test4_filenames.npy')

        # download test labels
        if not os.path.exists(os.path.join("data", "test4_labels.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test4_labels.npy',
                              'test4_labels.npy')

        # download and unzip validation data
        if not os.path.exists(os.path.join("data", "cv4_data.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv4_data.zip', 'cv4_data.zip')

        # download validation labels
        if not os.path.exists(os.path.join("data", "cv4_labels.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv4_labels.npy','cv4_labels.npy')

        if not os.path.exists(os.path.join("data", "cv4_filenames.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv4_filenames.npy',
                              'cv4_filenames.npy')

    elif what == 1:
        # download main training tfrecords files
        if not os.path.exists(os.path.join("data", "training_0.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training_0.tfrecords',
                              'training_0.tfrecords')

        if not os.path.exists(os.path.join("data", "training_1.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training_1.tfrecords',
                              'training_1.tfrecords')

        if not os.path.exists(os.path.join("data", "training_2.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training_2.tfrecords',
                              'training_2.tfrecords')

        if not os.path.exists(os.path.join("data", "training_3.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training_3.tfrecords',
                              'training_3.tfrecords')
    elif what == 0:
        # download MIAS test data
        if not os.path.exists(os.path.join("data", "mias_test_images.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/all_mias_slices.npy',
                              'mias_test_images.npy')

        if not os.path.exists(os.path.join("data", "mias_test_labels_enc.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/all_mias_labels.npy',
                              'mias_test_labels_enc.npy')

    elif what == 5:
        # download and unzip tfrecords training data
        if not os.path.exists(os.path.join("data", "training5_0.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training5_0.zip',
                              'training5_0.zip')

        if not os.path.exists(os.path.join("data", "training5_1.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training5_1.zip',
                              'training5_1.zip')

        if not os.path.exists(os.path.join("data", "training5_2.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training5_2.zip',
                              'training5_2.zip')

        if not os.path.exists(os.path.join("data", "training5_3.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training5_3.zip',
                              'training5_3.zip')

        if not os.path.exists(os.path.join("data", "training5_4.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training5_4.zip',
                              'training5_4.zip')

        # download and unzip test data
        if not os.path.exists(os.path.join("data", "test5_data.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test5_data.zip',
                              'test5_data.zip')

        if not os.path.exists(os.path.join("data", "test5_filenames.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test5_filenames.npy',
                              'test5_filenames.npy')

        # download test labels
        if not os.path.exists(os.path.join("data", "test5_labels.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test5_labels.npy',
                              'test5_labels.npy')

        # download and unzip validation data
        if not os.path.exists(os.path.join("data", "cv5_data.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv5_data.zip',
                              'cv5_data.zip')

        # download validation labels
        if not os.path.exists(os.path.join("data", "cv5_labels.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv5_labels.npy',
                              'cv5_labels.npy')

        if not os.path.exists(os.path.join("data", "cv5_filenames.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv5_filenames.npy',
                              'cv5_filenames.npy')

    elif what ==6:
        # download and unzip tfrecords training data
        if not os.path.exists(os.path.join("data", "training6_0.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training6_0.zip',
                              'training6_0.zip')

        if not os.path.exists(os.path.join("data", "training6_1.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training6_1.zip',
                              'training6_1.zip')

        if not os.path.exists(os.path.join("data", "training6_2.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training6_2.zip',
                              'training6_2.zip')

        if not os.path.exists(os.path.join("data", "training6_3.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training6_3.zip',
                              'training6_3.zip')

        if not os.path.exists(os.path.join("data", "training6_4.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training6_4.zip',
                              'training6_4.zip')

        # download and unzip test data
        if not os.path.exists(os.path.join("data", "test6_data.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test6_data.zip',
                              'test6_data.zip')

        if not os.path.exists(os.path.join("data", "test6_filenames.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test6_filenames.npy',
                              'test6_filenames.npy')

        # download test labels
        if not os.path.exists(os.path.join("data", "test6_labels.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test6_labels.npy',
                              'test6_labels.npy')

        # download and unzip validation data
        if not os.path.exists(os.path.join("data", "cv6_data.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv6_data.zip',
                              'cv6_data.zip')

        # download validation labels
        if not os.path.exists(os.path.join("data", "cv6_labels.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv6_labels.npy',
                              'cv6_labels.npy')

        if not os.path.exists(os.path.join("data", "cv6_filenames.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv6_filenames.npy',
                              'cv6_filenames.npy')

## Load the training data and return a list of the tfrecords file and the size of the dataset
## Multiple data sets have been created for this project, which one to be used can be set with the type argument
def get_training_data(what=5):
    if what == 5:
        train_path_10 = os.path.join("data", "training5_0.tfrecords")
        train_path_11 = os.path.join("data", "training5_1.tfrecords")
        train_path_12 = os.path.join("data", "training5_2.tfrecords")
        train_path_13 = os.path.join("data", "training5_3.tfrecords")
        train_path_14 = os.path.join("data", "training5_4.tfrecords")

        train_files = [train_path_10, train_path_11, train_path_12, train_path_13, train_path_14]
        total_records = 39316

    elif what == 4:
        train_path_10 = os.path.join("data", "training4_0.tfrecords")
        train_path_11 = os.path.join("data", "training4_1.tfrecords")
        train_path_12 = os.path.join("data", "training4_2.tfrecords")
        train_path_13 = os.path.join("data", "training4_3.tfrecords")
        train_path_14 = os.path.join("data", "training4_4.tfrecords")

        train_files = [train_path_10, train_path_11, train_path_12, train_path_13, train_path_14]
        total_records = 41527

    elif what == 6:
        train_path_10 = os.path.join("data", "training6_0.tfrecords")
        train_path_11 = os.path.join("data", "training6_1.tfrecords")
        train_path_12 = os.path.join("data", "training6_2.tfrecords")
        train_path_13 = os.path.join("data", "training6_3.tfrecords")
        train_path_14 = os.path.join("data", "training6_4.tfrecords")

        train_files = [train_path_10, train_path_11, train_path_12, train_path_13, train_path_14]
        total_records = 62764

    else:
        train_path_0 = os.path.join("data", "training_0.tfrecords")
        train_path_1 = os.path.join("data", "training_1.tfrecords")
        train_path_2 = os.path.join("data", "training_2.tfrecords")
        train_path_3 = os.path.join("data", "training_3.tfrecords")

        train_files = [train_path_0, train_path_1, train_path_2, train_path_3]
        total_records = 27393

    return train_files, total_records

def evaluate_model():
    pass

## functions to help build the graph
def _conv2d_batch_norm(input, filters, kernel_size=(3,3), stride=(1,1), training = tf.placeholder(dtype=tf.bool, name="is_training"), epsilon=1e-8, padding="SAME", seed=None, lambd=0.0, name=None):
    with tf.name_scope('layer_'+name) as scope:
        conv = tf.layers.conv2d(
            input,
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=seed),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambd),
            name='conv_'+name
        )

        # apply batch normalization
        conv = tf.layers.batch_normalization(
            conv,
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

        # apply relu
        conv = tf.nn.relu(conv, name='relu_'+name)

    return conv

def _dense_batch_norm(input, units,  training = tf.placeholder(dtype=tf.bool, name="is_training"), epsilon=1e-8, seed=None, dropout_rate=0.5, lambd=0.0, name=None):
    with tf.name_scope('layer_' + name) as scope:
        fc = tf.layers.dense(
            input,  # input
            units,  # 1024 hidden units
            activation=None,  # None
            kernel_initializer=tf.variance_scaling_initializer(scale=2, seed=seed),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambd),
            name="fc_"+name
        )

        fc = tf.layers.batch_normalization(
            fc,
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
            name='bn_fc_' + name
        )

        fc = tf.nn.relu(fc, name='fc_relu' + name)

        # dropout
        fc = tf.layers.dropout(fc, rate=dropout_rate, seed=seed, training=training)

    return fc

## load weights from a checkpoint, excluding any or including specified vars and returning initializer function
def load_weights(model_name, exclude=None, include=None):
    model_path = os.path.join("model", model_name + ".ckpt")

    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=exclude, include=include)
    init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

    return init_fn

def flatten(l):
    out = []
    for item in l:
        if isinstance(item, (list, tuple)):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out