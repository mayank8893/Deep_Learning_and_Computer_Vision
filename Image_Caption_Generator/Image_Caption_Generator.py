#!/usr/bin/env python
# coding: utf-8

# # Image caption Generator
# 
# The objective of this project is to **generate a caption for an image**. The dataset used consists of around 8000 images with 5 caption for each image. The features are extracted from both the image and the text captions for the input. The features will be concatenated to predict the next word of the caption. **CNN is used for image and LSTM is used for text. BLEU Score is used as a metric to evaluate the performance of the trained model.**
# 
# The dataset can be found here: https://www.kaggle.com/datasets/adityajn105/flickr8k.

# In[42]:


# import modules.
import pandas as pd
import os
import pickle
import numpy as np
from tqdm.notebook import tqdm # UI for data processing

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add


# In[2]:


# unzipping the zipped file
import zipfile as zf
files = zf.ZipFile("archive.zip", 'r')
files.extractall('/Users/ellietripathi/Python_Projects/Image_Caption_Generator/')
files.close()


# In[43]:


BASE_DIR = '/Users/ellietripathi/Python_Projects/Image_Caption_Generator/archive'
WORKING_DIR = '/Users/ellietripathi/Python_Projects/Image_Caption_Generator'


# ### Extracting Image features.

# In[44]:


# load vgg16 model. We will use VGG16 to extract features from the image.
model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
print(model.summary())


# In[9]:


# extract features from the image
features = {}
directory = os.path.join(BASE_DIR,'Images')

for img_name in tqdm(os.listdir(directory)):
    img_path = directory + '/' + img_name
    image = load_img(img_path, target_size = (224,224))
    # convert image pixels to numpy array.
    image = img_to_array(image)
    #reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # preprocess image for VGG
    image = preprocess_input(image)
    #extract features
    feature = model.predict(image, verbose=0)
    # get image ID
    image_id = img_name.split('.')[0]
    # store feature
    features[image_id] = feature


# In[10]:


# store features in pickle, so we dont have to extract feature again.
pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))


# In[46]:


# load features from pickle
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)


# ### Loading the Captions data

# In[47]:


with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()


# In[ ]:





# In[48]:


# creating mapping between image and caption
mapping = {}

for line in tqdm(captions_doc.split('\n')):
    # split the line by ','
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # remove .jpg extension from image_id
    image_id = image_id.split('.')[0]
    # converting the caption list into one string
    caption = " ".join(caption)
    #create a list of captions
    if image_id not in mapping:
        mapping[image_id] = []
    #store the caption
    mapping[image_id].append(caption)


# In[49]:


len(mapping)


# ### PreProcess Text Data

# In[50]:


def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            # preprocessing steps
            # convert to lower case
            caption = caption.lower()
            #delete digits, special characters, etal.
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            captions[i] = caption


# In[51]:


# looking at one caption
mapping['1000268201_693b08cb0e']


# In[52]:


# preprocess the text.
clean(mapping)


# In[53]:


# caption after preprocessing
mapping['1000268201_693b08cb0e']


# In[54]:


all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)


# In[55]:


len(all_captions)


# In[56]:


# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1


# In[57]:


vocab_size


# In[58]:


# get the maximum length of the caption
# useful for padding the sequence.
max_length = max(len(caption.split()) for caption in all_captions)
max_length


# ### Train Test Split

# In[59]:


image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]


# In[60]:


# create data generator to get data in batch (avoids session crash)
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    # loop over images
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            # process each caption
            for caption in captions:
                # encode the sequence
                seq = tokenizer.texts_to_sequences([caption])[0]
                # split the sequence into X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pairs
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    # store the sequences
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield [X1, X2], y
                X1, X2, y = list(), list(), list()
                n = 0


# ### Model Creation

# In[61]:


# encoder model
# image feature layers
inputs1 = Input(shape=(4096,)) # 4096 is the output of the VGG model.
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
# sequence feature layers
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# decoder model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# plot the model
plot_model(model, show_shapes=True)


# ![Screen%20Shot%202023-08-23%20at%201.35.55%20PM.png](attachment:Screen%20Shot%202023-08-23%20at%201.35.55%20PM.png)

# In[62]:


# train the model.
epochs = 20
batch_size = 32
steps = len(train) // batch_size

for i in range(epochs):
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    # fit for one epoch
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)


# In[65]:


#save the model.
model.save('best_model.h5')


# ### Generate captions for the image.

# In[66]:


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# In[67]:


# generate caption for an image.
# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text


# In[68]:


from nltk.translate.bleu_score import corpus_bleu
# validate with test data
actual, predicted = list(), list()

for key in tqdm(test):
    # get actual caption
    captions = mapping[key]
    # predict the caption for image
    y_pred = predict_caption(model, features[key], tokenizer, max_length) 
    # split into words
    actual_captions = [caption.split() for caption in captions]
    y_pred = y_pred.split()
    # append to the list
    actual.append(actual_captions)
    predicted.append(y_pred)
    
# calcuate BLEU score
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))


# ### Visualize the Results

# In[69]:


from PIL import Image
import matplotlib.pyplot as plt
def generate_caption(image_name):
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Images", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    # predict the caption
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)


# In[70]:


generate_caption("1001773457_577c3a7d70.jpg")


# In[76]:


generate_caption("1057089366_ca83da0877.jpg")


# In[77]:


generate_caption("132489044_3be606baf7.jpg")


# In[ ]:




