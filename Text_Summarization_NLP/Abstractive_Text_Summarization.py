#!/usr/bin/env python
# coding: utf-8

# # Text Summarization
# 
# In this project, I will work on creating an **abstractive text summarization** from an input text using **torch and transformer model T5-small**. 
# 
# Text summarization is a **natural language processing (NLP)** task that involves generating concise and coherent summaries from longer pieces of text, such as articles, documents, or web pages. The goal of text summarization is to distill the main ideas, key points, and essential information from the source text while preserving its meaning and context. Summarization has various applications, including information retrieval, content summarization for news articles, automatic document summarization, and more. 
# 
# **Extractive summarization** methods select and extract existing sentences or phrases directly from the source text to create a summary. These selected sentences are typically the most important or representative ones.
# 
# **Abstractive summarization** methods generate new sentences that capture the essence of the source text while using their own wording. These methods rely on natural language generation techniques and can paraphrase and combine information to create a more concise and coherent summary.

# In[103]:


# import dependencies.
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config


# In[104]:


model = T5ForConditionalGeneration.from_pretrained('t5-small') # initialize a pretrained model.
tokenizer = T5Tokenizer.from_pretrained('t5-small') # using a pretrained tokenizer.
device = torch.device('cpu') # cpu, gpu etc


# In[105]:


# Define the larger source text to be summarized
source_text = """
The invention of the internet has brought about revolutionary changes in the way we communicate, access information, and conduct business. 
It has transformed various aspects of our lives, from the way we shop to the way we socialize. The internet has also enabled the rapid exchange of knowledge, 
making education and research more accessible than ever before.

In recent years, social media platforms have become a significant part of online interactions. People use platforms such as Facebook, Twitter, and Instagram to connect 
with friends, share updates, and engage in discussions. However, the rise of social media has also raised concerns about privacy, misinformation, and its impact on mental health.

E-commerce, fueled by the internet, has revolutionized the retail industry. Consumers can now shop for products from around the world and have them delivered to their doorstep. 
Online marketplaces like Amazon have become giants in the e-commerce space, offering a wide range of products and services.

The internet has also transformed entertainment and media consumption. Streaming platforms like Netflix, YouTube, and Spotify allow users to access a vast library of content 
on-demand. Traditional media outlets have had to adapt to the digital age by offering online news and streaming services.

In the field of healthcare, the internet has enabled telemedicine, allowing patients to consult with doctors remotely. Medical information and resources are readily available 
online, empowering individuals to take charge of their health.

While the internet has brought many benefits, it has also given rise to challenges such as cybersecurity threats, online harassment, and the spread of fake news. 
Efforts are being made to address these issues and create a safer online environment.

In conclusion, the internet has had a profound impact on society, transforming the way we communicate, learn, shop, and more. It has opened up new opportunities and 
challenges that continue to shape our digital world.
"""


# In[106]:


# Tokenize and encode the source text
inputs = tokenizer.encode("summarize: " + source_text, return_tensors="pt", max_length=1024, truncation=True)


# In[107]:


# Looking at the input text
t5_input_text


# In[108]:


len(t5_input_text.split())


# In[109]:


# tokenize the text.
tokenized_text = tokenizer.encode(t5_input_text, return_tensors='pt', max_length=1024, truncation = True).to(device)


# In[110]:


# Generate the summary
summary_ids = model.generate(inputs, max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

# Decode the generated summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# In[111]:


# Print the original source text and the generated summary
print("Source Text:")
print(source_text)
print("\nGenerated Summary:")
print(summary)


# In[ ]:





# In[ ]:





# In[ ]:




