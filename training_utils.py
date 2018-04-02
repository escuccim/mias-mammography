import numpy as np
import os
import wget
from sklearn.cross_validation  import train_test_split
import tensorflow as tf

## download a file to a location in the data folder
def download_file(url, name):
    print("\nDownloading " + name + "...")

    # check that the data directory exists
    try:
        os.stat("data")
    except:
        os.mkdir("data")

    fname = wget.download(url, os.path.join('data', name))


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
def read_and_decode_single_example(filenames, label_type='label_normal', normalize=False, num_epochs=None):
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

    #tf.cast(image, dtype=tf.float32)

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
    X_cv, _, y_cv, _ = train_test_split(X_cv, y_cv, test_size=1 - percentage)

    return X_cv, y_cv


## evaluate the model to see the predictions
def evaluate_model(graph, config):
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

    if not os.path.exists(os.path.join("data", "test_labels.npy")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test_labels.npy', 'test_labels.npy')

    if not os.path.exists(os.path.join("data", "test_data.npy")):
        _ = download_file('https://s3.eu-central-1.amazonaws.com/aws.skoo.ch/files/test_data.npy', 'test_data.npy')
