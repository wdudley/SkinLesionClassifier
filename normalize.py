import cv2
import numpy as np
import os

dir_ = "/home/SharedStorage2/NewUsersDir/aledhari/wdudley/sp/datasets/ds2/illum_norm_ds/"
dir_tar = "/home/SharedStorage2/NewUsersDir/aledhari/wdudley/sp/datasets/ds2/illum_norm_color_norm/"

def illumination_norm(image):
  # read input
  hh, ww = image.shape[:2]
  #print(hh, ww)
  max_ = max(hh, ww)
  
  # illumination normalize
  ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
  
  # separate channels
  y, cr, cb = cv2.split(ycrcb)
  
  # get background which paper says (gaussian blur using standard deviation 5 pixel for 300x300 size image)
  # account for size of input vs 300
  sigma = int(5 * max_ / 512)
  #print('sigma: ',sigma)
  gaussian = cv2.GaussianBlur(y, (0, 0), sigma, sigma)
  
  # subtract background from Y channel
  y = (y - gaussian + 100)
  
  # merge channels back
  ycrcb = cv2.merge([y, cr, cb])
  
  #convert to BGR
  output = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
  
  return output
  
def color_norm(image):
  new_image = np.zeros((image.shape))
  output = cv2.normalize(image,  new_image, 0, 255, cv2.NORM_MINMAX)
  
  return output
  
def run_illumin():
  for root, subdirectories, files in os.walk(dir_):
    for file_name in files:
      path = root + "/" + file_name
      #print(path)
      if(".jpg" in path):
        image = cv2.imread(path)
        img_norm = illumination_norm(image)
      
        cv2.imwrite(dir_tar + file_name, img_norm)
        
def run_color_norm():
  for root, subdirectories, files in os.walk(dir_):
    for file_name in files:
      path = root + "/" + file_name
      #print(path)
      if(".jpg" in path):
        image = cv2.imread(path)
        img_norm = color_norm(image)
      
        cv2.imwrite(dir_tar + file_name, img_norm)
      

      
#run_illumin()
run_color_norm()
