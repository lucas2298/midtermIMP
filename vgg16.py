from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential, Model, optimizers
from keras.layers import Dropout, Flatten, Dense, Input

import os

# Run with GPU
import tensorflow as tf
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.compat.v1.Session(config=config) 
tf.compat.v1.keras.backend.set_session(sess)

# Path to the model weights files
weights_path = './input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
top_model_weights_path = ''
# Dimensions of our images
img_width, img_height = 224, 224

train_data_dir = './input/data/train'
validation_data_dir = './input/data/validation'

# Input tensor
input_tensor = Input(shape=(img_height, img_width, 3))

# Build the VGG16 network
model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

# Build a classifier model to put on top of the conv model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0,5))
top_model.add(Dense(1, activation='sigmoid'))

# Load fully-trained
# top_model.load_weights(top_model_weights_path)

model = Model(inputs = model.input, outputs = top_model(model.output))

# set the first 15 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:15]:
    layer.trainable = False

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr = 1e-4, momentum=0.9),
              metrics=['accuracy'])

# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.adam(lr = 1e-4),
#               metrics=['accuracy'])

# prepare data augmentation configuration

batch_size = 15

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# Train model

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size + 1
STEP_SIZE_VALI = validation_generator.n//validation_generator.batch_size + 1
print(validation_generator.n)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VALI,
    verbose=2
)

model.save('model.h5')

import pickle
pickle.dump(history.history, open('log', "wb"))