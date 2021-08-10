#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf


# In[3]:


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# In[4]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])


# In[5]:


predictions = model(x_train[:1]).numpy()
predictions


# In[6]:


tf.nn.softmax(predictions).numpy()


# In[7]:


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# In[8]:


loss_fn(y_train[:1], predictions).numpy()


# In[9]:


model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


# In[10]:


model.fit(x_train, y_train, epochs=5)


# In[11]:


model.evaluate(x_test,  y_test, verbose=2)


# In[12]:


probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])


# In[13]:


probability_model(x_test[:5])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




