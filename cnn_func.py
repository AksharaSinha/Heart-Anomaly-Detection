# -*- coding: utf-8 -*-
"""Untitled16.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16knUGXQhUgfEms7_YyV5ME9mtcZ2-zaV
"""

import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tqdm import tqdm

def cnn_function(xtrain,xtest,ytrain,ytest,encoders,cweights):

  optimiser_list = ["Nadam","Adamax","Adam"] 
  model_name_list = ["nadam_model","adamax_model","adam_model"]
  for i in tqdm(range(3)):
    print('\n',optimiser_list[i], "optimiser is currently running...")
    model = Sequential()
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, input_shape=(xtrain.shape[1],xtrain.shape[2],xtrain.shape[3]),padding="same", activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=2,padding="same", activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=2, padding="same", activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=2, padding="same", activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    model.add(Dense(len(encoders.classes_), activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimiser_list[i])
    model_name_list[i] = model.fit(xtrain, ytrain,
              batch_size=256,
              epochs=350,
              class_weight=cweights,
              validation_data=(xtest, ytest),
              shuffle=True, verbose=0)
  adam_loss = model_name_list[2].history["val_loss"]
  adamax_loss = model_name_list[1].history["val_loss"]
  nadam_loss = model_name_list[0].history["val_loss"]
  val_losses = []
  val_losses.append(nadam_loss)
  val_losses.append(adamax_loss)
  val_losses.append(adam_loss)

  adam_acc = model_name_list[2].history["val_accuracy"]
  adamax_acc = model_name_list[1].history["val_accuracy"]
  nadam_acc = model_name_list[0].history["val_accuracy"]
  val_accs = []
  val_accs.append(nadam_acc)
  val_accs.append(adamax_acc)
  val_accs.append(adam_acc)
  return val_losses, val_accs