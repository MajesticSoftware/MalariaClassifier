#!/usr/bin/env python
# coding: utf-8

# # Malaria Classifier

# In[1]:


import tensorflow as tf
import cv2
import numpy as np
import random
import os
import random
import string


# In[2]:


from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img


# In[3]:


# img0 = load_img('/Users/jeffreyscruggs/Desktop/A.I. Projects/cell_images/Parasitized/C33P1thinF_IMG_20150619_120645a_cell_216.png', target_size=(28, 28, 3))
img0 = load_img('./cell_images/Parasitized/C189P150ThinF_IMG_20151203_141308_cell_73.png', target_size=(28, 28, 3))

target_size=(28, 28, 3)


# In[4]:


img0


# In[5]:


img_array = img_to_array(img0)


# In[6]:


img_array.shape


# In[7]:


import glob


# In[8]:


# file_path = '/Users/jeffreyscruggs/Desktop/A.I. Projects/cell_images/Uninfected/'
file_path = './cell_images/Uninfected/'
f_names = glob.glob(file_path + '*.png')


# In[9]:


img_array.shape
print(f_names)


# In[10]:


imgs = []
for i in range(len(f_names)):  # f_names
    img = load_img(f_names[i], target_size=(28, 28, 3))  # read img
    arr_img = img_to_array(img)  # img to array
    arr_img = np.expand_dims(arr_img, axis=0)   # add one dim for batch
    imgs.append(arr_img) # append imgs to list
    print("loading no.%s image."%i)
x0 = np.concatenate([x for x in imgs]) # concatenate imgs


# In[11]:


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


# In[12]:


x0.shape


# In[ ]:





# In[13]:


y0=np.zeros(x0.shape[0])


# In[14]:


y0.shape


# In[15]:


# file_path = '/Users/jeffreyscruggs/Desktop/A.I. Projects/cell_images/Parasitized/'
file_path = './cell_images/Parasitized/'
f_names = glob.glob(file_path + '*.png')


# In[16]:


imgs = []
for i in range(len(f_names)):  #
    img = load_img(f_names[i], target_size=(28, 28, 3))  #
    arr_img = img_to_array(img)  # 图片转换为数组
    arr_img = np.expand_dims(arr_img, axis=0)   #
    imgs.append(arr_img) #
    print("loading no.%s image."%i)


# In[17]:


x1 = np.concatenate([x for x in imgs]) # concatenate


# In[18]:


y1=np.ones(x1.shape[0])


# In[19]:


y1


# In[20]:


X = np.vstack([x0,x1])


# In[21]:


X.shape


# In[22]:


y = np.concatenate([y0,y1])


# In[23]:


y


# In[24]:



from sklearn.model_selection import train_test_split


# In[25]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[27]:


X_train = X_train / 255.0
X_test = X_test / 255.0


# In[28]:


# check the concept of epoch and batch
# X_train.shape[0] is the TOTAL number of train images

epochs_num = 20
batch_size = 100
steps_per_epoch_num = X_train.shape[0] // batch_size


# In[29]:


train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))


# In[30]:


# train_dataset = train_dataset.batch(batch_size=10)
# train_dataset = train_dataset.repeat(count=10)
train_dataset = train_dataset.batch(batch_size=epochs_num)
train_dataset = train_dataset.repeat(count=epochs_num) # repeat should >= epochs_num
train_dataset = train_dataset.shuffle(buffer_size=10)


# In[31]:


test_dataset = test_dataset.batch(batch_size=epochs_num)
# test_dataset = test_dataset.batch(batch_size=100)
#test_dataset = test_dataset.repeat(count=1)


# In[32]:


classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Conv2D(8, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,3)))
classifier.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
classifier.add(tf.keras.layers.Flatten())
classifier.add(tf.keras.layers.Dense(64, activation='relu'))
classifier.add(tf.keras.layers.Dense(1, activation='sigmoid'))


# In[33]:


classifier.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics = [tf.keras.metrics.Accuracy()])


# In[34]:


classifier.summary()


# In[35]:


classifier.fit(
        train_dataset,
        steps_per_epoch= steps_per_epoch_num,
        epochs=epochs_num,
        validation_data = test_dataset
)


# In[ ]:


print(tf.__version__)


# In[ ]:





# In[ ]:




