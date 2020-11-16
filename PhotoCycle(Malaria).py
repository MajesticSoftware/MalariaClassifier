#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


model = tf.keras.models.load_model('malariaClasifier2')


# In[3]:


import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageFont, ImageDraw


# In[4]:


test_image = image.load_img('/Users/jeffreyscruggs/Desktop/A.I._Projects/cell_images/Parasitized/C39P4thinF_original_IMG_20150622_105253_cell_95.png', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print(result)


# In[5]:


print("0 = Parasitized, 1 = Uninfected")


# In[6]:


import cv2
import glob


# In[7]:


#Set Library Directory
path = "/Users/jeffreyscruggs/Desktop/DataSets/Shuffle/*.*"


# In[8]:


#Cycle through directory while predicting!
import time
for file in glob.glob(path):
    print(file)
    a = cv2.imread(file)
    test_image = image.load_img(file, target_size = (64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    print(result)
    
    if result == 0:
        print("This is a Parasitized Malaria cell.")
        img = Image.open(file)
        #Drawing Image
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/Users/jeffreyscruggs/Desktop/DataSets/Sketchzone.otf", 25)
    
        draw.text((32,32), "I", font = font)
    
        
        img.show(a)
        
        #Wait a 2 seconds, (Each 1000 equals 1 seconds, unit is in milliseconds.)
    
        k = cv2.waitKey(2000)
        k
        time.sleep(3)
        #Destroy Windows to save memory.
        cv2.destroyAllWindows()

    
    else:
        print("This is a Uninfected Malaria cell.")
        img = Image.open(file)
        #Drawing Image
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/Users/jeffreyscruggs/Desktop/DataSets/Sketchzone.otf", 25)
    
        draw.text((32,32), "U", font = font)
        img.show(a)
    
        #Wait a 2 seconds, (Each 1000 equals 1 seconds, unit is in milliseconds.)
    
        k = cv2.waitKey(2000)
        k
        time.sleep(3)
        #Destroy Windows to save memory.
        cv2.destroyAllWindows()


# In[ ]:




