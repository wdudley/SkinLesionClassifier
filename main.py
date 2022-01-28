import dataset as ds
import model
import tensorflow as tf
from datetime import datetime

train_csv = "/home/SharedStorage2/NewUsersDir/aledhari/wdudley/sp/training_labels.csv"
validation_csv = "/home/SharedStorage2/NewUsersDir/aledhari/wdudley/sp/validation_labels.csv"
testing_csv = "/home/SharedStorage2/NewUsersDir/aledhari/wdudley/sp/testing_labels.csv"
curr_dir = "/home/SharedStorage2/NewUsersDir/aledhari/wdudley/sp/datasets/ds2/"

num_classes = 4
input_shape = (512, 512, 3)
ds_dir = "illum_norm_color_norm"

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

ds3 = ds.Dataset(dataset_dirs = curr_dir + ds_dir, csv_dirs = [train_csv, validation_csv, testing_csv])

#ds3.display_class_hist()
class_weights = ds3.calc_class_weights()

train_gen, validation_gen, test_gen = ds3.load_dataset()

run_model = True

model_2 = model.ConvModel(num_classes = num_classes, train_set = train_gen, validation_set = validation_gen)
now = datetime.now()

def save_summary():
  from contextlib import redirect_stdout

  with open('figs/modelsummary-' + str(now) + '.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

if run_model:
  model_2.build_model_4()
  model_2.compile_model_2()
  
  model_2.model_summary()
  model_2.fit_model(num_epochs = 10, class_weights = None)
  model_2.test_model(test_data = test_gen)
  
  model_2.show_loss_graph()
  model_2.show_acc_graph()
  
  model_2.save_weights("weight_saves/model_4-" + str(now) + ".h5")
  save_summary()

if not run_model:
  model_2.test_model(test_data = test_gen, weights = "model_4illum_norm_color_norm.h5")
  save_summary()
  
