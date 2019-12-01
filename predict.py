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

# restored_model = load_model("model.h5")
model = load_model('model.h5')
# Input tensor
img_width, img_height = 150, 150


S = 224

directory = os.listdir("./input/data/validation/cats")
train_data_dir = './input/data/train'
validation_data_dir = './input/data/validation'
nb_train_samples = 7003
nb_validation_samples = 1002
epochs = 1
batch_size = 16

test_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary', shuffle=False)

score = model.evaluate_generator(validation_generator, nb_validation_samples/batch_size, workers=12, verbose=1)

scores = model.predict_generator(validation_generator, nb_validation_samples/batch_size, workers=12, verbose=1)

correct_cats = 0
correct_dogs = 0

x, y = 0, 0

for i, n in enumerate(validation_generator.filenames):
    if n.startswith("cats") and scores[i][0] <= 0.5:
        correct_cats += 1
    elif n.startswith("cats"):
        x+=1

    if n.startswith("dogs") and scores[i][0] > 0.5:
        correct_dogs += 1
    elif n.startswith("dogs"):
        y+=1

print("Incorrect cats:", x)
print("Incorrect dogs:", y)

print("Correct cats:", correct_cats, " Total: ", len(validation_generator.filenames))
print("Correct dogs:", correct_dogs, " Total: ", len(validation_generator.filenames))
print("Loss: ", score[0], "Accuracy: ", score[1])


# x = 0
# y = 0
# directory = os.listdir("./input/data/test")
# from keras.preprocessing.image import load_img, img_to_array
# from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
# for i in range(0, 10):
#     image = load_img('./input/data/test/'+directory[i], target_size=(S, S))
#     plt.imshow(image)
#     image = img_to_array(image)
#     image = image * [1./255]
#     image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
#     pred = model.predict(image)
#     print(pred[0][0])
#     plt.show()
#     if (pred <= 0.5):
#         x+=1
#     else:
#         y+=1
# print(x, y)

# import pickle
# log = pickle.load(open('log', "rb"))

# plt.plot(log['accuracy'],'green',label='Accuracy')
# plt.plot(log['loss'],'red',label='Loss')
# plt.legend(loc='upper left')
# plt.title('Training Accuracy & Loss')
# plt.xlabel('Epoch')
# plt.figure()
# plt.plot(log['val_accuracy'],'green',label='Accuracy')
# plt.plot(log['val_loss'],'red',label='Loss')
# plt.legend(loc='upper left')
# plt.title('Validation Accuracy & Loss')
# plt.xlabel('Epoch')
# plt.show()