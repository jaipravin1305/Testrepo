#!/usr/bin/env python
# coding: utf-8

# # 1. VGG16 / VGG19

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
import os

import warnings
warnings.filterwarnings("ignore")


# In[2]:


train_path = "C:/Users/jaipr/Downloads/seg_train/seg_train"
test_path = "C:/Users/jaipr/Downloads/seg_test/seg_test"


# In[3]:


# The number of classes of dataset
numberOfClass = len(glob(train_path + "/*"))
print("Number Of Class: ", numberOfClass)


# In[5]:


img = load_img(train_path + "//buildings//9938.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()


# In[6]:


image_shape = img_to_array(img)
print(image_shape.shape)


# In[7]:


train_data = ImageDataGenerator().flow_from_directory(train_path, target_size = (224,224))
test_data = ImageDataGenerator().flow_from_directory(test_path, target_size = (224,224))


# # Visualization

# In[9]:


for i in os.listdir(train_path):
    for j in os.listdir(os.path.join(train_path, i)):
        img = load_img(os.path.join(train_path, i, j))
        plt.imshow(img)
        plt.title(i)
        plt.axis("off")
        plt.show()
        break


# In[10]:


vgg16 = VGG16()


# In[11]:


# Layers of vgg16 
vgg16.summary()


# In[12]:


# layers of vgg16
vgg16_layer_list = vgg16.layers
for i in vgg16_layer_list:
    print(i)


# In[13]:


# add the layers of vgg16 in my created model.
vgg16Model = Sequential()
for i in range(len(vgg16_layer_list)-1):
    vgg16Model.add(vgg16_layer_list[i])


# In[14]:


# the final version of the model
vgg16Model.summary()


# In[15]:


# Close the layers of vgg16
for layers in vgg16Model.layers:
    layers.trainable = False


# In[16]:


# Last layer
vgg16Model.add(Dense(numberOfClass, activation = "softmax"))


# In[17]:


# After I added last layer in created model.
vgg16Model.summary()


# In[18]:


# I create compile part.
vgg16Model.compile(loss = "categorical_crossentropy",
             optimizer = "rmsprop",
             metrics = ["accuracy"])


# # Training Model

# In[19]:


# Traning with model
batch_size = 32

hist_vgg16 = vgg16Model.fit_generator(train_data, 
                                      steps_per_epoch = 1600 // batch_size, 
                                      epochs = 10, 
                                      validation_data = test_data, 
                                      validation_steps = 800 // batch_size)


# In[20]:


vgg16Model.save_weights("deneme.h5")


# In[21]:


plt.plot(hist_vgg16.history["loss"], label = "training loss")
plt.plot(hist_vgg16.history["val_loss"], label = "validation loss")
plt.legend()
plt.show()


# In[22]:


plt.plot(hist_vgg16.history["accuracy"], label = "accuracy")
plt.plot(hist_vgg16.history["val_accuracy"], label = "validation accuracy")
plt.legend()
plt.show()


# In[23]:


import json, codecs
with open("deneme.json","w") as f:
    json.dump(hist_vgg16.history, f)


# In[24]:


with codecs.open("./deneme.json","r", encoding = "utf-8") as f:
    load_result = json.loads(f.read())


# In[25]:


load_result


# In[26]:


plt.plot(load_result["loss"], label = "training loss")
plt.plot(load_result["val_loss"], label = "validation loss")
plt.legend()
plt.show()


# In[27]:


# Accuracy And Validation Accuracy
plt.plot(load_result["accuracy"], label = "accuracy")
plt.plot(load_result["val_accuracy"], label = "validation accuracy")
plt.legend()
plt.show()


# In[28]:


# Import VGG19 model
vgg19 = VGG19()


# In[29]:


vgg19.summary()


# In[30]:


# Layers of vgg19 
vgg19_layer_list = vgg19.layers
for i in vgg19_layer_list:
    print(i)


# In[31]:


vgg19Model = Sequential()
for i in range(len(vgg19_layer_list)-1):
    vgg19Model.add(vgg19_layer_list[i])


# In[32]:


# Finish version of my created model.
vgg19Model.summary()


# In[33]:


# Close the layers of vgg16
for layers in vgg19Model.layers:
    layers.trainable = False


# In[34]:


vgg19Model.add(Dense(numberOfClass, activation = "softmax"))


# In[35]:


vgg19Model.summary()


# In[36]:


vgg19Model.compile(loss = "categorical_crossentropy",
             optimizer = "rmsprop",
             metrics = ["accuracy"])


# In[37]:


# Training with my created model
hisy_vgg19 = vgg19Model.fit_generator(train_data,
                               steps_per_epoch = 1600 // batch_size,
                               epochs = 10,
                               validation_data = test_data,
                               validation_steps = 800 // batch_size)


# In[38]:


# Loss And Validation Loss
plt.plot(hisy_vgg19.history["loss"], label = "training loss")
plt.plot(hisy_vgg19.history["val_loss"], label = "validation loss")
plt.legend()
plt.show()


# In[39]:


# Accuracy And Validation Accuracy
plt.plot(hisy_vgg19.history["accuracy"], label = "accuracy")
plt.plot(hisy_vgg19.history["val_accuracy"], label = "validation accuracy")
plt.legend()
plt.show()


# # 2. Classification using InceptionV3 model

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as layers


# In[2]:


labels=pd.read_csv('monkey_labels.txt')


# In[3]:


labels


# # Step 1: Pre-process and create train set

# In[5]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('C:/Users/jaipr/training/training',
                                                 target_size = (128,128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# # Step 2: pre-process and create test set

# In[6]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = train_datagen.flow_from_directory('C:/Users/jaipr/validation/validation',
                                                 target_size = (128,128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# In[8]:


import IPython.display as ipd

ipd.Image('C:/Users/jaipr/training/training/n5/n5021.jpg')


# # Step 3: Import the pre- trained model

# In[9]:


from tensorflow.keras.applications.inception_v3 import InceptionV3
base_model = InceptionV3(input_shape = (128, 128, 3), include_top = False, weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False


# # Step 4: Add Flattening, hidden and output layers

# In[10]:


x=base_model.output
x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(10, activation='sigmoid')(x)

inception = tf.keras.models.Model(base_model.input, x)
inception.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[11]:


inception.summary()


# # Step 5: Train the model

# In[12]:


Inception_hist=inception.fit(training_set, validation_data=test_set, epochs=20)


# # Step 6: Train and Test accuracy, loss plots

# In[13]:


# summarize history for accuracy
plt.plot(Inception_hist.history['accuracy'])
plt.plot(Inception_hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(Inception_hist.history['loss'])
plt.plot(Inception_hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:




