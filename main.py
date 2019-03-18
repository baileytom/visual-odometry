import scipy
import tensorflow as tf
import pathlib
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import IPython.display as display
import collections
import bisect
import imageio
import numpy as np
from tensorflow.python.keras import utils
import cv2

# Preprocessing, the best part of any project

# Get the images
data_root = pathlib.Path('./data')
image_root = data_root / 'rgb'
image_paths = list(image_root.glob('*.png'))
image_paths = sorted([str(path) for path in image_paths])
image_count = len(image_paths)

# Get the labels (filenames) and data
f = open(data_root / 'groundtruth.txt', 'r')
motion_data = f.readlines()
f = open(data_root / 'rgb.txt', 'r')
label_data = f.readlines()

# Cut off the metadata
label_data = label_data[3:]
motion_data = motion_data[3:]

# Why I had to do the following:
#
# The dataset gives us N frames, but the position data does not correspond to a frame.
# They are collected seperately so there are 4x as many lines for position as there are
# frames. So to get the position at a given frame, we have to basically estimate based
# on the closest timestamp.
# =========================================
pat = collections.OrderedDict()
for datum in motion_data:
    pat[float(datum.split()[0])] = np.array([float(n) for n in datum.split()[1:]])

def position_at_time(timestamp):
    ind = bisect.bisect_left(list(pat.keys()), timestamp)
    return list(pat.items())[ind][1]

positions = []
for label in label_data:
    timestamp = float(label.split()[0])
    closest_position = position_at_time(timestamp)
    positions.append(closest_position)
# =======================================

## image processing
def image_to_np(filename):
    image = cv2.imread(filename)
    image = cv2.resize(image, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
    return image
    
images = [image_to_np(f) for f in image_paths]


img_x, img_y = 32, 32
input_shape = (img_x, img_y, 3)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        32,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation='relu',
        input_shape=input_shape
    ),
    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ),
    tf.keras.layers.Flatten(
        input_shape=input_shape
    ),
    tf.keras.layers.Dense(7)
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])


images = np.array(images)
positions = np.array(positions)

model.fit(images, positions, epochs=100)

img = np.expand_dims(images[0], axis=0)

print(model.predict(img))
print(positions[0])


#loss, acc = model.evaluate(x_test, y_test)
