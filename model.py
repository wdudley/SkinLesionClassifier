import dataset as ds
import custom_model_builders
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, BatchNormalization, MaxPooling2D, GlobalMaxPooling2D, Dropout, GlobalAveragePooling2D
# from tensorflow.keras.applications import EfficientNetB0 as Net
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Model 
from tensorflow import keras
from keras import regularizers, optimizers
from datetime import datetime
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import classification_report

# TODO: Add more hyperparams, make model selection easier to test
class ConvModel:
  def __init__ (self, num_classes, kernel_size = 3, train_set = [], validation_set = []):
    self.train_set = train_set
    self.num_classes = num_classes
    self.validation_set = validation_set
    self.model = None
    self.hist = None
    
  def build_model_1(self):
    self.model = Sequential()
    self.model.add(Conv2D(2, kernel_size=(13, 13),
                 activation='relu'))
    self.model.add(Flatten())
    self.model.add(Dense(self.num_classes, activation='softmax'))
  
  # Deeper model
  # Results in worse overfitting
  # Needs more regularization, decreases in depth and height or more data
  def build_model_2(self):
  
    self.model = Sequential()
    self.model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu'))
    self.model.add(Conv2D(32, (3, 3), activation='relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Conv2D(64, (1, 1), activation='relu'))
    self.model.add(Conv2D(64, (3, 3), activation='relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Conv2D(64, (1, 1), activation='relu'))
    self.model.add(Conv2D(64, (3, 3), activation='relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Flatten())
    self.model.add(Dense(64, activation='relu'))
    self.model.add(Dropout(0.5))
    self.model.add(Dense(32, activation='relu'))
    self.model.add(Dropout(0.5))
    self.model.add(Dense(self.num_classes, activation='softmax'))
    
  # Resnet50 backbone, imagenet weights
  def build_model_3(self):
    resnet = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(512,512,3))
    
    self.model = Sequential()
    self.model.add(resnet)
    self.model.add(GlobalAveragePooling2D())
    self.model.add(Dropout(0.5))
    self.model.add(BatchNormalization())
    self.model.add(Dense(self.num_classes, activation='softmax'))
    
  def build_model_4(self):
    from efficientnet.keras import EfficientNetB1 as Net
    from efficientnet.keras import center_crop_and_resize, preprocess_input
    
    conv_base = Net(weights="imagenet", include_top=False, input_shape=(512, 512, 3,), classes = 4)
    
    self.model = Sequential()
    self.model.add(conv_base)
    self.model.add(GlobalMaxPooling2D(name="gap"))
    self.model.add(Dense(32, activation='relu'))
    self.model.add(Dropout(0.2, name="dropout_1"))
    self.model.add(Dense(64, activation='relu'))
    self.model.add(Dropout(0.2, name="dropout_out"))
    self.model.add(Dense(4, activation='softmax', name="fc_out"))
    
    self.model.summary()
    
  def build_custom_model(self, params):
    self.model = Sequential()
  
    for layer_type in params:
      layer_type_ = layer_type[0]
      print(layer_type)
      if (layer_type_ == 'Conv2D'):
        if (len(layer_type) == 3):
          custom_model_builders.add_conv(self.model, layer_type[1], input_shape = layer_type[2])
        else:
          custom_model_builders.add_conv(self.model, layer_type[1])
      elif(layer_type_ == 'Pool'):
        custom_model_builders.add_pool(self.model, layer_type[1])
      elif(layer_type_ == 'Flatten'):
        custom_model_builders.add_flatten(self.model)
      elif(layer_type_ == 'Dense'):
        custom_model_builders.add_dense(self.model, layer_type[1])
      elif(layer_type_ == 'Dropout'):
        custom_model_builders.add_dropout(self.model, layer_type[1])
      elif(layer_type_ == 'Dense_Out'):
        custom_model_builders.add_dense_out(self.model, layer_type[1])
      else:
        print("Failed to detect layer type: ", layer_type_)
    
  def compile_model(self):
    self.model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss='CategoricalCrossentropy',
      metrics=['accuracy'])
      
    #self.model.summary()
      
  def compile_model_2(self):
    self.model.compile(
      loss='categorical_crossentropy',
      optimizer=optimizers.RMSprop(lr=2e-5),
      metrics=['accuracy'])
      
  def model_summary(self):
    self.model.summary()
    
  def fit_model(self, num_epochs = 10, class_weights = None):
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    self.hist = self.model.fit(
    self.train_set,
    validation_data=self.validation_set,
    epochs=num_epochs,
    class_weight = class_weights,
    callbacks=[callback]
    )

  def test_model(self, test_data, weights = None):
    if weights != None:
      self.build_model_4()
      self.compile_model_2()
      self.load_weights(weights)
    
    Y_pred = self.model.predict_generator(test_data)
    y_pred = np.argmax(Y_pred,axis=1)
    
    #print(y_pred)
    
    print(confusion_matrix(test_data.classes, y_pred))
    print(classification_report(test_data.classes, y_pred))
  
  def show_loss_graph(self):
    now = datetime.now()
    plt.plot(self.hist.history['loss'])
    plt.plot(self.hist.history['val_loss'])
    plt.title('model loss / epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('figs/loss' + str(now) + '.png')
    plt.clf()
    
  def show_acc_graph(self):
    now = datetime.now()
    plt.plot(self.hist.history['accuracy'])
    plt.plot(self.hist.history['val_accuracy'])
    plt.title('model acc / epoch')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('figs/acc' + str(now) + '.png')
    plt.clf()
    
  def save_weights(self, save_name):
    self.model.save(save_name)
    
  def load_weights(self, save_name):
    self.model.load_weights(save_name)
  
class ConnectedModel:
  def __init__ (self, num_layers, layer_size_list):
    self.numlayers = num_layers
    self.layer_size_list = layer_size_list