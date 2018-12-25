import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor



class ImageDataGenerator(object):
    def __init__(self, csv_file, mode, batch_size, num_classes, shuffle=True,
                 buffer_size=1000):

        self.csv_file = csv_file
        self.num_classes = num_classes

        # retrieve the data from the text file
        self._read_csv_file()

        # number of samples in the dataset
        self.data_size = len(self.labels)

        # initial shuffling of the file and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        # create dataset
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train,
                            num_parallel_calls=1).prefetch(100*batch_size)

        elif mode == 'inference':
            data = data.map(self._parse_function_inference,
                            num_parallel_calls=1).prefetch(100*batch_size)
        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data

    def _read_csv_file(self):
        self.img_paths = []
        self.labels = []
        df = pd.read_csv(self.csv_file)
        self.img_paths = df['guid/image'].tolist()
        self.labels = df['label'].tolist()

    def _shuffle_lists(self):
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, filename, label):
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])

        return img_resized, one_hot

    def _parse_function_inference(self, filename, label):
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])

        return img_resized, one_hot
