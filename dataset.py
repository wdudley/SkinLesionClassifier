import os
import csv
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_csv(dataset_dir, csv_name):
  csv_list = []
  sub_dirs = os.listdir(dataset_dir)
  for x_dirs in sub_dirs:
    curr_dir = dataset_dir + '/' + x_dirs
    class_count = 0
    for y_dirs in os.listdir(curr_dir):
      for root, dirs, files in os.walk(curr_dir + '/' + y_dirs, topdown=False):
        for filename in files:
          if class_count == 0 or class_count == 1 or class_count == 2:
            lst = [filename, str(class_count)]
            csv_list.append(lst)
          elif class_count == 4:
            lst = [filename, str(4)]
            csv_list.append(lst)
      class_count += 1

  with open(csv_name, 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(csv_list)
    
def remove_undersampled(csv_label_file, new_csv_name, classes_to_remove):
  df = pd.read_csv(csv_label_file)
  
  for row, col in df.iterrows():
    if(col.Class == 3 or col.Class == 5 or col.Class == 6 or col.Class == 7):
      df = df.drop(row)
      
  df.to_csv(new_csv_name)
    
class Dataset:
  def __init__ (self, dataset_dirs, csv_dirs = [], class_weight = []):
    self.dataset_dirs = dataset_dirs
    self.csv_dirs = csv_dirs
    self.class_weight = class_weight
    self.dataframe = []
    
  def load_dataset(self):
    datagen=ImageDataGenerator(rescale=1./255.)
    
    dataframe_ = []
    
    for csv_ in self.csv_dirs:
      data = pd.read_csv(csv_, dtype=str)
      
      generator = datagen.flow_from_dataframe(
        dataframe = data,
        directory=self.dataset_dirs,
        x_col="Image",
        y_col="Class",
        color_mode="rgb",
        batch_size=32,
        shuffle=False,
        class_mode="categorical",
        image_size = (512, 512))
        
      dataframe_.append(generator)
      generator.reset()
     

    return dataframe_[0], dataframe_[1], dataframe_[2]
  
  def test_image_tensor(self, img_path):
    img = cv2.imread(img_path)
    cv2.imshow('name', img)
    #img = Image.open(img_path)
    #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #ImageShow.show(img)
  
  def load_dataset_from_dir(self):
    train_set = tf.keras.preprocessing.image_dataset_from_directory(
      self.dataset_dirs[0], #       validation_split=0.2,       subset = "training",
      color_mode = "grayscale",
      seed = 123,
      image_size = (224, 224),
      batch_size= 32
    )

    test_set = tf.keras.preprocessing.image_dataset_from_directory(
      self.dataset_dirs[1], #       validation_split=0.2,       subset = "validation",
      color_mode = "grayscale",
      seed = 123,
      image_size = (224, 224),
      batch_size= 32
    )
    
    return train_set, test_set
    
  def calc_class_weights(self):
    class_weights = {}
    df = pd.read_csv(self.csv_dirs[0])
    samples = df["Class"].value_counts()
    max_sample=np.max(samples)
    print (max_sample)
    for i in range (len(samples)):
      class_weights[i]=max_sample/samples[i]
    for key, value in class_weights.items():
      print ( key, ' : ', value)
      
    return class_weights
  
  def display_class_hist(self):
    for csv_ in self.csv_dirs:
      data = pd.read_csv(csv_, usecols = ["Class"], dtype=str)
      plt.hist(data)
    # plt.xlabel=('Class', 14)
    # plt.xlabel=('Population', 14)
      plt.show()
      plt.clf()
    
    