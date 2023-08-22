#!/usr/bin/env python
# coding: utf-8

# # Dog Classification Project.
# 
# In this project, I will try to classify a dog picture into one of the following five breeds: beagle, bernese mountain, doberman, labrador retriever and siberian husky. I will start with unzipping the file and then splitting the datset into a training and test set. Then I will visualize just one image. At this point, I will train a model  using **Convolutional Neural Networks and look at its validation accuracy**. 
# 
# I will then work on improving the model by adding more layers and by **data augmentation**. The best model performs at about **50% accuracy**. **This is beacuse it mainly mis-classifies the labrator retriever dataset beacuse they have both black and golden coat.** Finally I visualize the dog image, actual label and label predicted by the Neural netowrks.

# In[76]:


# unzipping the zipped file
import zipfile as zf
files = zf.ZipFile("images.zip", 'r')
files.extractall('/Users/ellietripathi/Python_Projects/Dog_Breed_Classification/')
files.close()


# ### Exploring the data

# In[77]:


breeds = ['.ipynb_checkpoints', 'beagle', 'bernese_mountain_dog', 'doberman', 'labrador_retriever', 'siberian_husky']


# In[78]:


#importing tensorflow.
import tensorflow as tf


# In[79]:


# standard arguments for the dataset.
args = {
    "labels":"inferred",
    "label_mode":"categorical",
    "batch_size": 32,
    "image_size":(256,256),
    "seed":1,
    "validation_split":0.2,
    "class_names": breeds
}


# In[80]:


# getting the train dataset.
train = tf.keras.utils.image_dataset_from_directory("images", subset="training", **args)


# In[81]:


# getting the test dataset.
test = tf.keras.utils.image_dataset_from_directory("images", subset="validation", **args)


# In[82]:


# train a batch element.
train


# In[83]:


# taking the first image from the batch.
first = train.take(1)
first


# In[84]:


# looking at the first image.
images, labels = list(first)[0]
first_image = images[0]
first_image[:3,:3,2]


# In[ ]:





# In[ ]:





# In[86]:


# plotting the first image.
from PIL import Image
Image.fromarray(first_image.numpy().astype("uint8"))


# In[87]:


labels[0]


# In[88]:


# makes performacy faster by loading data into memory.
train = train.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
test = test.cache().prefetch(buffer_size = tf.data.AUTOTUNE)


# # Model implementation
# 
# Starting with a very simple model.

# In[89]:


# making the model
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

model = Sequential([
    layers.Rescaling(1./255), # rescale data
    layers.Conv2D(16, 3, padding = "same", activation = "relu", input_shape = (256, 256, 3)),
    layers.Flatten(),
    layers.Dense(128, activation = "relu"),
    layers.Dense(len(breeds))
]
)


# In[90]:


# compile the model
model.compile(optimizer="adam", 
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
              metrics=["accuracy"])


# In[91]:


# fit the model
history = model.fit(train, validation_data = test, epochs = 5, verbose=1)


#  The model does not perform very well and it **overfits the training data**.

# In[93]:


# getting the model summary.
model.summary()


# In[94]:


# plotting the accuracy.
import pandas as pd

history_df = pd.DataFrame.from_dict(history.history)
history_df[["accuracy", "val_accuracy"]].plot()


# The model is **overfitting**. It works very well for the training data but not so for the validation data.

# In[95]:


# defining a function so that we dont have to use the same code again and again.
def train_model(network, epochs = 5):
    model = Sequential(network)
    model.compile(optimizer="adam", 
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
              metrics=["accuracy"])
    history = model.fit(train, validation_data = test, epochs = epochs)
    history_df = pd.DataFrame.from_dict(history.history)
    return history_df, model


# ### Improving the model

# In[96]:


# Making a better model by reducing verfitting using MaxPooling2D
network = [
    # rescale data
    layers.Rescaling(1./255),
    
    layers.Conv2D(16, 4, padding = "same", activation = "relu", input_shape = (256, 256, 3)),
    layers.MaxPooling2D(), # helps reduce overfitting
    layers.Conv2D(32, 4, padding = "same", activation = "relu", input_shape = (256, 256, 3)),
    layers.MaxPooling2D(), # helps reduce overfitting
    layers.Conv2D(64, 4, padding = "same", activation = "relu", input_shape = (256, 256, 3)),
    layers.MaxPooling2D(), # helps reduce overfitting
    
    layers.Dropout(0.2), 
    layers.Flatten(),
    
    layers.Dense(128, activation = "relu"),
    
    layers.Dense(len(breeds))
]

history_df, model = train_model(network, epochs = 10)


# In[97]:


# plotting the accuracy.
history_df[["accuracy", "val_accuracy"]].plot()


# The model hasnt improved a lot.
# ## Data Augmentation

# In[98]:


# creating more data by proxy.
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal", seed = 1),
    layers.RandomRotation(0.2, seed = 1),
    layers.RandomZoom(0.2, seed=1)
])


# In[99]:


# making the full network.
full_network = [data_augmentation] + network


# In[100]:


history_df, model = train_model(full_network, epochs = 10)


# In[101]:


#plotting accuracy.
history_df[["accuracy", "val_accuracy"]].plot()


# In[102]:


# making predictions.
preds = model.predict(test)


# In[103]:


# getting labels for predicted class.
import numpy as np
predicted_class = np.argmax(preds, axis=1)
predicted_class


# In[104]:


# getting labels for actual class.
actual_labels = np.concatenate([y for x,y in test], axis =0)
actual_class = np.argmax(actual_labels, axis =1)
actual_class


# In[105]:


# plotting the image, actual label and predicted label for dogs.
import itertools

actual_image = [x.numpy().astype("uint8") for x,y in test]
actual_image = list(itertools.chain.from_iterable(actual_image))
actual_image = [Image.fromarray(a) for a in actual_image]

pred_df = pd.DataFrame(zip(predicted_class, actual_class, actual_image), 
                       columns=["predictions", "actual", "image"])
pred_df["predictions"] = pred_df["predictions"].apply(lambda x: breeds[x])
pred_df["actual"] = pred_df["actual"].apply(lambda x: breeds[x])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[106]:


pred_df.head()


# In[107]:


import base64
import io

def image_formatter(img):
    with io.BytesIO() as buffer:
        img.save(buffer, 'png')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f'<img src="data:image/jpeg;base64,{img_str}">'

pred_df.head(10).style.format({'image': image_formatter})


# In[ ]:




