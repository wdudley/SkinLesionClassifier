import model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, BatchNormalization, MaxPooling2D, GlobalMaxPooling2D, Dropout, GlobalAveragePooling2D

def add_conv(model, layer_type_params, input_shape = None):
  for layer in layer_type_params:
    if input_shape:
      model.add(Conv2D(layer[0], (layer[1],layer[1]), input_shape = input_shape))
    else:
      model.add(Conv2D(layer[0], (layer[1],layer[1])))

def add_pool(model, pool_size_n):
  model.add(MaxPooling2D(pool_size=(pool_size_n, pool_size_n)))
  
def add_flatten(model):
  model.add(Flatten())
  
def add_dense(model, layer_type_params):
  for layer in layer_type_params:
    model.add(Dense(layer, activation='relu'))

def add_dropout(model, droput_rate):
  model.add(Dropout(droput_rate))
  
def add_dense_out(model, num_classes):
  model.add(Dense(num_classes, activation = 'softmax'))