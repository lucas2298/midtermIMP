from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential, Model
from keras.models import load_model
from keras.layers import Dropout, Flatten, Dense, Input

# Run with GPU
import tensorflow as tf
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.compat.v1.Session(config=config) 
tf.compat.v1.keras.backend.set_session(sess)
from matplotlib import pyplot as plt
import os
import cv2

import pickle
log = pickle.load(open('log', "rb"))

# plt.figure(figsize=(1.5, 50), dpi = 100)

plt.plot(log['accuracy'],'green',label='Accuracy')
plt.plot(log['loss'],'red',label='Loss')
plt.legend(loc='upper left')
plt.title('Training Accuracy & Loss')
plt.xlabel('Epoch')
plt.figure()
plt.plot(log['val_accuracy'],'green',label='Accuracy')
plt.plot(log['val_loss'],'red',label='Loss')
plt.legend(loc='upper left')
plt.title('Validation Accuracy & Loss')
plt.xlabel('Epoch')
plt.show()