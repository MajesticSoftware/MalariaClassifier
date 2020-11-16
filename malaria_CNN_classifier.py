#!/usr/bin/env python
# coding: utf-8

# In[33]:


import tensorflow as tf
import cv2
import numpy as np
import random
import os

import random
import string


# In[34]:


from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img


# In[36]:


img0 = load_img('cell_images/Uninfected/C1_thinF_IMG_20150604_104722_cell_115.png', target_size=(28, 28, 3))


# In[37]:


img0


# In[38]:


img_array = img_to_array(img0)


# In[40]:


img_array.shape


# In[10]:


import glob


# In[42]:


file_path = 'cell_images/Uninfected/'
f_names = glob.glob(file_path + '*.png')


# In[44]:


img_array.shape


# In[45]:


imgs = []
for i in range(len(f_names)):  # f_names
    img = load_img(f_names[i], target_size=(28, 28, 3))  # read img
    arr_img = img_to_array(img)  # img to array
    arr_img = np.expand_dims(arr_img, axis=0)   # add one dim for batch
    imgs.append(arr_img) # append imgs to list
    print("loading no.%s image."%i)
x0 = np.concatenate([x for x in imgs]) # concatenate imgs


# ##The input shape
# 
# What flows between layers are tensors. Tensors can be seen as matrices, with shapes.   
# 
# In Keras, the input layer itself is not a layer, but a tensor. It's the starting tensor you send to the first hidden layer. This tensor must have the same shape as your training data. 
# 
# **Example:** if you have 30 images of 50x50 pixels in RGB (3 channels), the shape of your input data is `(30,50,50,3)`. Then your input layer tensor, must have this shape (see details in the "shapes in keras" section).   
# 
# Each type of layer requires the input with a certain number of dimensions:
# 
# - `Dense` layers require inputs as `(batch_size, input_size)`       
# - 2D convolutional layers need inputs as:`(batch_size, img_height, img_width, channels)`    
# 

# In[50]:


x0.shape


# In[51]:


y0=np.zeros(x0.shape[0])


# In[53]:


y0


# In[54]:


file_path = 'cell_images/Parasitized//'
f_names = glob.glob(file_path + '*.png')


# In[56]:


imgs = []
for i in range(len(f_names)):  #
    img = load_img(f_names[i], target_size=(28, 28, 3))  #
    arr_img = img_to_array(img)  # 图片转换为数组
    arr_img = np.expand_dims(arr_img, axis=0)   #
    imgs.append(arr_img) #
    print("loading no.%s image."%i)


# In[57]:


x1 = np.concatenate([x for x in imgs]) # concatenate


# In[58]:


y1=np.ones(x1.shape[0])


# In[59]:


y1


# In[60]:


X = np.vstack([x0,x1])


# In[61]:


X.shape


# In[62]:


y = np.concatenate([y0,y1])


# In[63]:


y


# In[20]:


from sklearn.model_selection import train_test_split


# In[64]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[22]:


# https://numpy.org/doc/stable/reference/generated/numpy.zeros.html


# In[23]:


X_train = X_train / 255.0
X_test = X_test / 255.0


# In[ ]:


# Dataset api


# In[67]:


train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))


# ### batch

# In[71]:


train_dataset = train_dataset.batch(batch_size=10)
train_dataset = train_dataset.repeat(count=2)
train_dataset = train_dataset.shuffle(buffer_size=10)


# In[72]:


test_dataset = test_dataset.batch(batch_size=100)
test_dataset = test_dataset.repeat(count=1)


# In[73]:


# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28, 3)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(1, activation='sigmoid')   
# ])


# In[74]:


classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Conv2D(8, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,3)))
classifier.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
classifier.add(tf.keras.layers.Flatten())
classifier.add(tf.keras.layers.Dense(64, activation='relu'))
classifier.add(tf.keras.layers.Dense(1, activation='sigmoid'))


# ![image.png](attachment:image.png)

# In[75]:


# Compile model
classifier.compile(optimizer ='adam', loss = 'binary_crossentropy')


# In[76]:


classifier.summary()


# In[77]:


classifier.fit(
        train_dataset,
        steps_per_epoch=1000//100,
        epochs=20,
        validation_data = test_dataset
)


# In[ ]:




