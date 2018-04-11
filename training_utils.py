import numpy as np
import os
import wget
import zipfile
from sklearn.cross_validation  import train_test_split
import tensorflow as tf

## open zip files
def unzip(file, destination):
    foo = zipfile.ZipFile(file, mode='r', allowZip64=True)
    foo.extractall(destination)

    return True

## download a file to a location in the data folder
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
    except:
        print("Error downloading", url)



## Batch generator
def get_batches(X, y, batch_size, distort=True):
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

        yield X_return, y[batch_idx]


## read data from tfrecords file
def read_and_decode_single_example(filenames, label_type='label_normal', normalize=False, distort=False, num_epochs=None):
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

    if normalize:
        image = tf.image.per_image_standardization(image)

    # random flipping of image
    if distort:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

    #tf.cast(image, dtype=tf.float32)

    # return the image and the label
    return image, label


## load the test data from files
def load_validation_data(data="validation", how="class", percentage=0.5):
    if data == "validation":
        # load the two data files
        X_cv_0 = np.load(os.path.join("data", "cv_data_0.npy"))
        labels_0 = np.load(os.path.join("data", "cv_labels_0.npy"))

        X_cv_1 = np.load(os.path.join("data", "cv_data_1.npy"))
        labels_1 = np.load(os.path.join("data", "cv_labels_1.npy"))

        # concatenate them
        X_cv = np.concatenate([X_cv_0, X_cv_1], axis=0)
        labels = np.concatenate([labels_0, labels_1], axis=0)

        # delete the old files to save memory
        del (X_cv_0)
        del (X_cv_1)
        del (labels_0)
        del (labels_1)

    elif data == "test":
        X_te_0 = np.load(os.path.join("data", "test2_data_0.npy"))
        te_labels_0 = np.load(os.path.join("data", "test2_labels_0.npy"))

        X_te_1 = np.load(os.path.join("data", "test2_data_1.npy"))
        te_labels_1 = np.load(os.path.join("data", "test2_labels_1.npy"))

        # concatenate them
        X_cv = np.concatenate([X_te_0, X_te_1], axis=0)
        labels = np.concatenate([te_labels_0, te_labels_1], axis=0)

        # delete the old files to save memory
        del (X_te_0)
        del (X_te_1)
        del (te_labels_0)
        del (te_labels_1)

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
    X_cv, _, y_cv, _ = train_test_split(X_cv, y_cv, test_size=1 - percentage)

    return X_cv, y_cv


## evaluate the model to see the predictions
def evaluate_model(graph, config, model_name, how="normal", batch_size=32):
    X_te, y_te = load_validation_data(how=how, data="test")

    with tf.Session(graph=graph, config=config) as sess:
        # create the saver
        saver = tf.train.Saver()
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, './model/' + model_name + '.ckpt')

        test_accuracy = []
        test_recall = []
        test_predictions = []
        ground_truth = []

        # evaluate the test data
        for X_batch, y_batch in get_batches(X_te, y_te, batch_size, distort=False):
            yhat, test_acc_value, test_recall_value = sess.run([predictions, accuracy, rec_op], feed_dict=
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
    print("Mean Recall:", np.mean(test_recall))
    print("Mean Accuracy:", np.mean(test_accuracy))

    return test_accuracy, test_recall, test_predictions, ground_truth

## Download the data if it doesn't already exist
def download_data(what="new"):
    if what == "new":

        # download and unzip tfrecords training data
        if not os.path.exists(os.path.join("data", "training2_0.zip")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training2_0.zip',
                              'training2_0.zip')

        if not os.path.exists(os.path.join("data", "training2_1.zip")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training2_1.zip',
                              'training2_1.zip')

        if not os.path.exists(os.path.join("data", "training2_2.zip")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training2_2.zip',
                              'training2_2.zip')

        if not os.path.exists(os.path.join("data", "training2_3.zip")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training2_3.zip',
                              'training2_3.zip')

        if not os.path.exists(os.path.join("data", "training2_4.zip")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training2_4.zip',
                              'training2_4.zip')

        # download and unzip test data
        if not os.path.exists(os.path.join("data", "test2_data_0.zip")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test2_data_0.zip',
                              'test2_data_0.zip')

        if not os.path.exists(os.path.join("data", "test2_data_1.zip")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test2_data_1.zip',
                              'test2_data_1.zip')

        # download test labels
        if not os.path.exists(os.path.join("data", "test2_labels_0.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test2_labels_0.npy',
                              'test2_labels_0.npy')

        if not os.path.exists(os.path.join("data", "test2_labels_1.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test2_labels_1.npy',
                              'test2_labels_1.npy')

        # download and unzip validation data
        if not os.path.exists(os.path.join("data", "cv_data_0.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv_data_0.npy',
                              'cv_data_0.npy')

        if not os.path.exists(os.path.join("data", "cv_data_0.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv_data_0.npy',
                              'cv_data_0.npy')

        # download validation labels
        if not os.path.exists(os.path.join("data", "cv_labels_0.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv_labels_0.npy',
                              'cv_labels_0.npy')

        if not os.path.exists(os.path.join("data", "cv_labels_1.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/cv_labels_1.npy',
                              'cv_labels_1.npy')

    elif what == "old":
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

        # download secondary training tfrecords files
        if not os.path.exists(os.path.join("data", "training3_0.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training3_0.tfrecords',
                              'training3_0.tfrecords')

        if not os.path.exists(os.path.join("data", "training3_1.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training3_1.tfrecords',
                              'training3_1.tfrecords')

        if not os.path.exists(os.path.join("data", "training3_2.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training3_2.tfrecords',
                              'training3_2.tfrecords')

        if not os.path.exists(os.path.join("data", "training3_3.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training3_3.tfrecords',
                              'training3_3.tfrecords')

        if not os.path.exists(os.path.join("data", "training3_4.tfrecords")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/training3_4.tfrecords',
                              'training3_4.tfrecords')

        # download MIAS test data
        if not os.path.exists(os.path.join("data", "mias_test_images.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/mias_test_images.npy',
                              'mias_test_images.npy')

        if not os.path.exists(os.path.join("data", "mias_test_labels_enc.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/mias_test_labels_enc.npy',
                              'mias_test_labels_enc.npy')

        # download validation data
        if not os.path.exists(os.path.join("data", "test2_labels_0.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test2_labels_0.npy',
                              'test2_labels_0.npy')

        if not os.path.exists(os.path.join("data", "test2_labels_1.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test2_labels_1.npy',
                              'test2_labels_1.npy')

        if not os.path.exists(os.path.join("data", "test2_data_0.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test2_data_0.npy',
                              'test2_data_0.npy')

        if not os.path.exists(os.path.join("data", "test2_data_1.npy")):
            _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test2_data_1.npy',
                              'test2_data_1.npy')

def get_training_data(type="new"):
    if type == "new":
        train_path_10 = os.path.join("data", "training2_0.tfrecords")
        train_path_11 = os.path.join("data", "training2_1.tfrecords")
        train_path_12 = os.path.join("data", "training2_2.tfrecords")
        train_path_13 = os.path.join("data", "training2_3.tfrecords")
        train_path_14 = os.path.join("data", "training2_4.tfrecords")

        train_files = [train_path_10, train_path_11, train_path_12, train_path_13, train_path_14]
        total_records = 31027
    elif type == "medium":
        train_path_10 = os.path.join("data", "training3_0.tfrecords")
        train_path_11 = os.path.join("data", "training3_1.tfrecords")
        train_path_12 = os.path.join("data", "training3_2.tfrecords")
        train_path_13 = os.path.join("data", "training3_3.tfrecords")
        train_path_14 = os.path.join("data", "training3_4.tfrecords")

        train_files = [train_path_10, train_path_11, train_path_12, train_path_13, train_path_14]
        total_records = 23297
    else:
        train_path_0 = os.path.join("data", "training_0.tfrecords")
        train_path_1 = os.path.join("data", "training_1.tfrecords")
        train_path_2 = os.path.join("data", "training_2.tfrecords")
        train_path_3 = os.path.join("data", "training_3.tfrecords")

        train_files = [train_path_0, train_path_1, train_path_2, train_path_3]
        total_records = 27393

    return train_files, total_records