import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as ds
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

size = '224x224'
name = 'imagenet_resized/'+size
val_or_test = 'validation'
data_dir = 'dataset/datasets_tfrecord/'

dataset_train, info = ds.load(
    name, with_info=True, as_supervised=True, data_dir=data_dir, split='train')
dataset_validation = ds.load(
    name, as_supervised=True, data_dir=data_dir, split=val_or_test)
dataset_train = dataset_train.batch(128)
dataset_train = dataset_train.map(
    lambda x, y: (tf.cast(x, dtype=tf.float32), y))
dataset_validation = dataset_validation.batch(128)
dataset_validation = dataset_validation.map(
    lambda x, y: (tf.cast(x, dtype=tf.float32), y))


def getModel():
    return keras.Sequential([
        keras.layers.Conv2D(96, input_shape=(224, 224, 3), kernel_size=(
            11, 11), strides=4, activation='relu'),  # (
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=2),  # (13,13)
        keras.layers.BatchNormalization(),

        keras.ZeroPadding2D(),
        keras.layers.Conv3D(256, kernal_size=(5, 5, 48),
                            strides=1, activation='relu'),  # (13,13)
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=2),  # (6,6)
        keras.layers.BatchNormalization(),

        keras.layers.ZeroPadding2D(padding=(1, 1)),  # (8,8)
        keras.layers.Conv2D(384, 3, strides=1, activation='relu'),  # (6,6)

        keras.layers.ZeroPadding2D(padding=(1, 1)),  # (8,8)
        keras.layers.Conv2D(384, 3, strides=1, activation='relu'),  # (6,6)

        keras.layers.ZeroPadding2D(padding=(1, 1)),  # (8,8)
        keras.layers.Conv2D(256, 3, strides=1, activation='relu'),  # (6,6)
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),  # (3,3)

        #                              keras.layers.Conv2D(4096,2,activation='relu'), #(1,1)
        #                             keras.layers.Flatten(),
        keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dense(1000, activation='softmax')
    ])


model = getModel()
model.summary()
exit()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(dataset_train, epochs=100, validation_data=dataset_validation)
test_value = model.evaluate(dataset_validation, verbose=0)
print("Evaluated loss on test data: {:.4f}, Accuracy: {:.4f}".format(
    test_value[0], test_value[1]))
