import tensorflow as tf
import numpy as np
import tensorflow_datasets as ds
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'


@tf.function
def getDCTFromJPEG(img):
    print(img)


# size = '64x64'
# name = 'imagenet_resized/'+size
# val_or_test = 'validation'
# data_dir = '../../dataset/datasets_tfrecord/'


# resnet = tf.keras.applications.resnet50
if __name__ == "__main__":
    print(getDCTFromJPEG(tf.io.read_file('./test_16.jpeg')))