#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#image = mpimg.imread("chelsea-the-cat.png")


# In[ ]:


#Set Library Directory
path = "/Users/jeffreyscruggs/Desktop/DataSets/Shuffle/*.*"


# In[ ]:


#Cycle through directory while predicting!
import time
for file in glob.glob(path):
    print(file)
    a = mpimg.imread(file)
    test_image = image.load_img(file, target_size = (64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    print(result)
    
    if result == 0:
        print("This is a Parasitized Malaria cell.")
        img = Image.open(file)
        
    
        
        plt.imshow(a)
        plt.text(60, 60, 'Parisitized', bbox=dict(fill=False, edgecolor='red', linewidth=1))
        plt.show()
        
        #Wait a 1 seconds
        time.sleep(1)
        plt.close()
    else:
        print("This is a Uninfected Malaria cell.")
        img = Image.open(file)
        
        plt.imshow(a)
        plt.text(60, 60, 'Uninfected', bbox=dict(fill=False, edgecolor='blue', linewidth=1))
        plt.show()
        time.sleep(1)
        plt.close()
        #Destroy Windows to save memory.


# In[ ]:





# In[ ]:





# In[ ]:




