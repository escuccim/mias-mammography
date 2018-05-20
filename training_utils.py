import numpy as np
import os
import wget
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import math

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
                print("\nZip file extracted and deleted", name)
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

## Code for data augmentation for images and labels take from http://ddokkddokk.tistory.com/11
def _do_nothing(image, label):
    return image, label

def _random_true_false():
    prob = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    predicate = tf.less(prob, 0.5)
    return predicate

def _image_and_label_flip(image, label):
    image_flip = tf.image.flip_left_right(image)
    label_flip = tf.image.flip_left_right(label)
    return image_flip, label_flip

def _image_random_flip(image, label):
    predicate = _random_true_false()
    image_aug, label_aug = tf.cond(predicate, lambda: _image_and_label_flip(image, label), lambda: _do_nothing(image, label))
    return image_aug, label_aug

## read data from tfrecords file
def read_and_decode_single_example(filenames, label_type='label_normal', normalize=False, distort=False, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)

    reader = tf.TFRecordReader()

    if label_type != 'label':
        label_type = 'label_' + label_type

    _, serialized_example = reader.read(filename_queue)
    if label_type != 'label_mask':
        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'label_normal': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string)
            })

        # extract the data
        label = features[label_type]
        image = tf.decode_raw(features['image'], tf.uint8)

        # reshape and scale the image
        image = tf.reshape(image, [299, 299, 1])

        # random flipping of image
        if distort:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)

    else:
        features = tf.parse_single_example(
            serialized_example,
            features={
                # We know the length of both fields. If not the
                # tf.VarLenFeature could be used
                'label': tf.FixedLenFeature([], tf.string),
                'image': tf.FixedLenFeature([], tf.string)
            })

        label = tf.decode_raw(features['label'], tf.uint8)
        image = tf.decode_raw(features['image'], tf.uint8)

        label = tf.cast(label, tf.int32)
        # image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        image = tf.reshape(image, [288, 288, 1])
        label = tf.reshape(label, [288, 288, 1])

        # if distort:
        #     image, label = _image_random_flip(image, label)

    if normalize:
        image = tf.image.per_image_standardization(image)

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
        elif which == 8:
            X_cv = np.load(os.path.join("data", "cv8_data.npy"))
            labels = np.load(os.path.join("data", "cv8_labels.npy"))
        elif which == 9:
            X_cv = np.load(os.path.join("data", "cv9_data.npy"))
            labels = np.load(os.path.join("data", "cv9_labels.npy"))
        elif which == 10:
            X_cv = np.load(os.path.join("data", "cv10_data.npy"))
            labels = np.load(os.path.join("data", "cv10_labels.npy"))
        elif which == 11:
            X_cv = np.load(os.path.join("data", "cv11_data.npy"))
            labels = np.load(os.path.join("data", "cv11_labels.npy"))
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
        elif which == 8:
            X_cv = np.load(os.path.join("data", "test8_data.npy"))
            labels = np.load(os.path.join("data", "test8_labels.npy"))
        elif which == 9:
            X_cv = np.load(os.path.join("data", "test9_data.npy"))
            labels = np.load(os.path.join("data", "test9_labels.npy"))
        elif which == 10:
            X_cv = np.load(os.path.join("data", "test10_data.npy"))
            labels = np.load(os.path.join("data", "test10_labels.npy"))
        elif which == 11:
            X_cv = np.load(os.path.join("data", "test11_data.npy"))
            labels = np.load(os.path.join("data", "test11_labels.npy"))
    elif data == "mias":
        if which == 9:
            X_cv = np.load(os.path.join("data", "all_mias_slices9.npy"))
            labels = np.load(os.path.join("data", "all_mias_labels9.npy"))
        else:
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
    elif how == "mask":
        y_cv = labels.astype(np.int32)

    # shuffle the data
    X_cv, y_cv = shuffle(X_cv, y_cv)

    return X_cv, y_cv

## Download the data if it doesn't already exist, many datasets have been created, which one to download can be specified using
## the what argument
def download_data(what=4):
    if what == 8:
        # download and unzip tfrecords training data
        if not os.path.exists(os.path.join("data", "training8_0.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training8_0.zip',
                              'training8_0.zip')

        if not os.path.exists(os.path.join("data", "training8_1.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training8_1.zip',
                              'training8_1.zip')

        if not os.path.exists(os.path.join("data", "training8_2.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training8_2.zip',
                              'training8_2.zip')

        if not os.path.exists(os.path.join("data", "training8_3.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training8_3.zip',
                              'training8_3.zip')

        if not os.path.exists(os.path.join("data", "training8_4.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training8_4.zip',
                              'training8_4.zip')

        # download and unzip test data
        if not os.path.exists(os.path.join("data", "test8_data.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test8_data.zip',
                              'test8_data.zip')

        if not os.path.exists(os.path.join("data", "test8_filenames.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test8_filenames.npy',
                              'test8_filenames.npy')

        # download test labels
        if not os.path.exists(os.path.join("data", "test8_labels.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test8_labels.npy',
                              'test8_labels.npy')

        # download and unzip validation data
        if not os.path.exists(os.path.join("data", "cv8_data.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv8_data.zip', 'cv8_data.zip')

        # download validation labels
        if not os.path.exists(os.path.join("data", "cv8_labels.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv8_labels.npy','cv8_labels.npy')

        if not os.path.exists(os.path.join("data", "cv8_filenames.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv8_filenames.npy',
                              'cv8_filenames.npy')

    if what == 9:
        # download and unzip tfrecords training data
        if not os.path.exists(os.path.join("data", "training9_0.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training9_0.zip',
                              'training9_0.zip')

        if not os.path.exists(os.path.join("data", "training9_1.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training9_1.zip',
                              'training9_1.zip')

        if not os.path.exists(os.path.join("data", "training9_2.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training9_2.zip',
                              'training9_2.zip')

        if not os.path.exists(os.path.join("data", "training9_3.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training9_3.zip',
                              'training9_3.zip')

        if not os.path.exists(os.path.join("data", "training9_4.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training9_4.zip',
                              'training9_4.zip')

        # download and unzip test data
        if not os.path.exists(os.path.join("data", "test9_data.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test9_data.zip',
                              'test9_data.zip')

        if not os.path.exists(os.path.join("data", "test9_filenames.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test9_filenames.npy',
                              'test9_filenames.npy')

        # download test labels
        if not os.path.exists(os.path.join("data", "test9_labels.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test9_labels.npy',
                              'test9_labels.npy')

        # download and unzip validation data
        if not os.path.exists(os.path.join("data", "cv9_data.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv9_data.zip', 'cv9_data.zip')

        # download validation labels
        if not os.path.exists(os.path.join("data", "cv9_labels.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv9_labels.npy','cv9_labels.npy')

        if not os.path.exists(os.path.join("data", "cv9_filenames.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv9_filenames.npy',
                              'cv9_filenames.npy')
    elif what == 10:
        # download and unzip tfrecords training data
        if not os.path.exists(os.path.join("data", "training10_0.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training10_0.zip',
                              'training10_0.zip')

        if not os.path.exists(os.path.join("data", "training10_1.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training10_1.zip',
                              'training10_1.zip')

        if not os.path.exists(os.path.join("data", "training10_2.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training10_2.zip',
                              'training10_2.zip')

        if not os.path.exists(os.path.join("data", "training10_3.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training10_3.zip',
                              'training10_3.zip')

        if not os.path.exists(os.path.join("data", "training10_4.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training10_4.zip',
                              'training10_4.zip')

        # download and unzip test data
        if not os.path.exists(os.path.join("data", "test10_data.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test10_data.zip',
                              'test10_data.zip')

        # download test labels
        if not os.path.exists(os.path.join("data", "test10_labels.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test10_labels.npy',
                              'test10_labels.npy')

        # download and unzip validation data
        if not os.path.exists(os.path.join("data", "cv10_data.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv10_data.zip', 'cv10_data.zip')

        # download validation labels
        if not os.path.exists(os.path.join("data", "cv10_labels.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv10_labels.npy','cv10_labels.npy')
    
    elif what == 11:
        # download and unzip tfrecords training data
        if not os.path.exists(os.path.join("data", "training11_0.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training11_0.zip',
                              'training11_0.zip')

        if not os.path.exists(os.path.join("data", "training11_1.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training11_1.zip',
                              'training11_1.zip')

        if not os.path.exists(os.path.join("data", "training11_2.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training11_2.zip',
                              'training11_2.zip')

        if not os.path.exists(os.path.join("data", "training11_3.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training11_3.zip',
                              'training11_3.zip')

        if not os.path.exists(os.path.join("data", "training11_4.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training11_4.zip',
                              'training11_4.zip')

        # download and unzip test data
        if not os.path.exists(os.path.join("data", "test11_data.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test11_data.zip',
                              'test11_data.zip')

        # download test labels
        if not os.path.exists(os.path.join("data", "test11_labels.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test11_labels.zip',
                              'test11_labels.zip')

        # download and unzip validation data
        if not os.path.exists(os.path.join("data", "cv11_data.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv11_data.zip', 'cv11_data.zip')

        # download validation labels
        if not os.path.exists(os.path.join("data", "cv11_labels.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv11_labels.zip','cv11_labels.zip')
    
    elif what == 0:
        # download MIAS test data
        if not os.path.exists(os.path.join("data", "mias_test_images.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/all_mias_slices.npy',
                              'mias_test_images.npy')

        if not os.path.exists(os.path.join("data", "mias_test_labels_enc.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/all_mias_labels.npy',
                              'mias_test_labels_enc.npy')

        # download MIAS test data
        if not os.path.exists(os.path.join("data", "all_mias_slices9.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/all_mias_slices9.npy',
                              'all_mias_slices9.npy')

        if not os.path.exists(os.path.join("data", "all_mias_labels9.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/all_mias_labels9.npy',
                              'all_mias_labels9.npy')

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
    if what == 6:
        train_path_10 = os.path.join("data", "training6_0.tfrecords")
        train_path_11 = os.path.join("data", "training6_1.tfrecords")
        train_path_12 = os.path.join("data", "training6_2.tfrecords")
        train_path_13 = os.path.join("data", "training6_3.tfrecords")
        train_path_14 = os.path.join("data", "training6_4.tfrecords")

        train_files = [train_path_10, train_path_11, train_path_12, train_path_13, train_path_14]
        total_records = 62764

    elif what == 8:
        train_path_10 = os.path.join("data", "training8_0.tfrecords")
        train_path_11 = os.path.join("data", "training8_1.tfrecords")
        train_path_12 = os.path.join("data", "training8_2.tfrecords")
        train_path_13 = os.path.join("data", "training8_3.tfrecords")
        train_path_14 = os.path.join("data", "training8_4.tfrecords")

        train_files = [train_path_10, train_path_11, train_path_12, train_path_13, train_path_14]
        total_records = 40559

    elif what == 9:
        train_path_10 = os.path.join("data", "training9_0.tfrecords")
        train_path_11 = os.path.join("data", "training9_1.tfrecords")
        train_path_12 = os.path.join("data", "training9_2.tfrecords")
        train_path_13 = os.path.join("data", "training9_3.tfrecords")
        train_path_14 = os.path.join("data", "training9_4.tfrecords")

        train_files = [train_path_10, train_path_11, train_path_12, train_path_13, train_path_14]
        total_records = 43739

    elif what == 10:
        train_path_10 = os.path.join("data", "training10_0.tfrecords")
        train_path_11 = os.path.join("data", "training10_1.tfrecords")
        train_path_12 = os.path.join("data", "training10_2.tfrecords")
        train_path_13 = os.path.join("data", "training10_3.tfrecords")
        train_path_14 = os.path.join("data", "training10_4.tfrecords")

        train_files = [train_path_10, train_path_11, train_path_12, train_path_13, train_path_14]
        total_records = 55890

    elif what == 11:
        train_path_10 = os.path.join("data", "training11_0.tfrecords")
        train_path_11 = os.path.join("data", "training11_1.tfrecords")
        train_path_12 = os.path.join("data", "training11_2.tfrecords")
        train_path_13 = os.path.join("data", "training11_3.tfrecords")
        train_path_14 = os.path.join("data", "training11_4.tfrecords")

        train_files = [train_path_10, train_path_11, train_path_12, train_path_13, train_path_14]
        total_records = 33241
    else:
        raise ValueError('Invalid dataset!')

    return train_files, total_records

def evaluate_model():
    pass

## functions to help build the graph
def _conv2d_batch_norm(input, filters, kernel_size=(3,3), stride=(1,1), training = tf.placeholder(dtype=tf.bool, name="is_training"), epsilon=1e-8, padding="SAME", seed=None, lambd=0.0, name=None, activation="relu"):
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

        if activation == "relu":
            # apply relu
            conv = tf.nn.relu(conv, name='relu_'+name)
        elif activation == "elu":
            conv = tf.nn.elu(conv, name="elu_" + name)

    return conv

def _dense_batch_norm(input, units,  training = tf.placeholder(dtype=tf.bool, name="is_training"), epsilon=1e-8, activation="relu", seed=None, dropout_rate=0.5, lambd=0.0, name=None):
    with tf.name_scope('fc_' + name) as scope:
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

        if activation == "elu":
            fc = tf.nn.elu(fc, name="fc_elu" + name)
        elif activation == None:
            pass
        else:
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

def _scale_input_data(X, contrast=None, mu=104.1353, scale=255.0):
    # if we are adjusting contrast do that
    if contrast and contrast != 1.0:
        X_adj = tf.image.adjust_contrast(X, contrast)
    else:
        X_adj = X

    # cast to float
    if X_adj.dtype != tf.float32:
        X_adj = tf.cast(X_adj, dtype=tf.float32)

    # center the pixel data
    X_adj = tf.subtract(X_adj, mu, name="centered_input")

    # scale the data
    X_adj = tf.divide(X_adj, scale)

    return X_adj

# Function to do the data augmentation on the GPU instead of the CPU, doing it on the CPU significantly slowed down training
# Taken from https://becominghuman.ai/data-augmentation-on-gpu-in-tensorflow-13d14ecf2b19
def augment(images, labels,
            horizontal_flip=False,
            vertical_flip=False,
            augment_labels=False,
            mixup=0):  # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf

    # My experiments showed that casting on GPU improves training performance
    if images.dtype != tf.float32:
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)

    with tf.name_scope('augmentation'):
        shp = tf.shape(images)
        batch_size, height, width = shp[0], shp[1], shp[2]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        # The list of affine transformations that our image will go under.
        # Every element is Nx8 tensor, where N is a batch size.
        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        if horizontal_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if vertical_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if transforms:
            images = tf.contrib.image.transform(
                images,
                tf.contrib.image.compose_transforms(*transforms),
                interpolation='BILINEAR')  # or 'NEAREST'

            if augment_labels:
                labels = tf.contrib.image.transform(
                    labels,
                    tf.contrib.image.compose_transforms(*transforms),
                    interpolation='BILINEAR')  # or 'NEAREST'

        def cshift(values):  # Circular shift in batch dimension
            return tf.concat([values[-1:, ...], values[:-1, ...]], 0)

        if mixup > 0:
            beta = tf.distributions.Beta(mixup, mixup)
            lam = beta.sample(batch_size)
            ll = tf.expand_dims(tf.expand_dims(tf.expand_dims(lam, -1), -1), -1)
            images = ll * images + (1 - ll) * cshift(images)
            labels = lam * labels + (1 - lam) * cshift(labels)

    return images, labels

def standardize(tensor):
    # cast to float 32
    if tensor.dtype != tf.float32:
        tensor = tf.image.convert_image_dtype(tensor, dtype=tf.float32)

    standardized_tensor = tf.div(
        tf.subtract(
            tensor,
            tf.reduce_min(tensor, axis=0)
        ),
        tf.subtract(
            tf.reduce_max(tensor, axis=0),
            tf.reduce_min(tensor, axis=0)
        )
    )
    return standardized_tensor

def plot_results(y_, yhat, x_, threshold=20):
    for i in range(len(yhat)):
        if (np.sum(yhat[i] == 1) > threshold) or (np.sum(y_[i] == 1) > threshold):
            f, ax = plt.subplots(1, 3, figsize=(10, 4))
            ax[0].imshow(x_[i].reshape(288,288))
            ax[0].set_title("Image")
            ax[1].imshow(y_[i].reshape(288,288))
            ax[1].set_title("Label")
            ax[2].imshow(yhat[i].reshape(288,288))
            ax[2].set_title("Prediction")
            plt.show()