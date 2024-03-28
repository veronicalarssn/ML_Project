#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential


# In[ ]:


pip install torch torchvision


# In[ ]:


pip install tensorflow


# In[2]:


#Re-Load the dataset
loaded_data = np.load("my_dataset_aug_full.npz", allow_pickle=True)

# Access images and labels
loaded_images = loaded_data['images']

# Now you can use loaded_images and loaded_labels in your code
image_df=pd.DataFrame(loaded_images)
image_df.head()


# In[3]:


#split it into features and labels
features = image_df.iloc[:, :51529]
labels = image_df.iloc[:, 51530]
labels_coded=image_df.iloc[:,51531]
labels_coded


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(features, labels_coded, test_size=0.2, random_state=42)

# Display the shapes of the resulting splits
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[ ]:


one_pic = X_train.iloc[0]
one_pic_array = one_pic.values.astype(float)  # Convert to float
one_pic_image = one_pic_array.reshape((227, 227))

plt.figure()
plt.imshow(one_pic_image)
plt.colorbar()
plt.grid(False)
plt.show()


# The data must be preprocessed before training the network. When inspecting one image in the training set, it can be observed that the pixel values fall in the range of 0 to 255.
# These values can be scaled to a range of 0 to 1 before feeding them to the neural network model. To do so, divide the values by 255. It's important that the training set and the testing set be preprocessed in the same way:

# In[ ]:


X_train=X_train/255.0
X_test=X_test/255.0


# In[ ]:


#to verify whether it worked: plot
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show()


# Building the neural network requires configuring the layers of the model, then compiling the model.
# 
# Set up the layers
# The basic building block of a neural network is the layer. Layers extract representations from the data fed into them. Hopefully, these representations are meaningful for the problem at hand.
# 
# Most of deep learning consists of chaining together simple layers. Most layers, such as tf.keras.layers.Dense, have parameters that are learned during training.
# 
# The first layer in this network, tf.keras.layers.Flatten, transforms the format of the images from a two-dimensional array (of 227 by 227 pixels) to a one-dimensional array (of 227 * 227 = 51528 pixels). Think of this layer as unstacking rows of pixels in the image and lining them up. This layer has no parameters to learn; it only reformats the data. --< true???
# 
# After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers. These are densely connected, or fully connected, neural layers. The first Dense layer has 128 nodes (or neurons). The second (and last) layer returns a logits array with length of 27. Each node contains a score that indicates the current image belongs to one of the 27 classes.

# ## MODEL 1

# In[5]:


model_1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(227,227)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(27)
])


# In[6]:


model_1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[ ]:


#To start training, call the model.fit method—so called because it "fits" the model to the training data:
model_1.fit(X_train, y_train, epochs=10)


# In[ ]:


#Next, compare how the model performs on the test dataset:
test_loss, test_acc = model_1.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)


# In[ ]:


#predictions
probability_model = tf.keras.Sequential([model_1, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(X_test)
print(predictions[0])
#A prediction is an array of 10 numbers. They represent the model's "confidence" that the image corresponds to each of the 27 different logos.
#You can see which label has the highest confidence value by calling:
print(f'most probable logo is',np.argmax(predictions[0]))


# In[ ]:


#the model is most confident that this image is XX
#Examining the test label shows that this classification is correct?False?:

y_test[0]


# In[ ]:


#https://www.tensorflow.org/tutorials/keras/classification
#https://www.tensorflow.org/tutorials/images/cnn


# ## 2ND MODEL 

# In[ ]:


model_2 = Sequential()
model_2.add(keras.layers.Flatten(input_shape=227,227))
model_2.add(keras.layers.Dense(100, activation='relu'))
model_2.add(kersa.layers.Dropout(0.2))
model_2.add(keras.layers.Dense(27)


# In[ ]:


model_2.summary()


# In[ ]:


predictions = model_2(X_train[:1]).numpy()
predictions


# In[ ]:


#Using `tf.nn.softmax` function converts these logits to *probabilities* for each class: 
tf.nn.softmax(predictions).numpy()


# In[ ]:


#Defining a loss function for training using `losses.SparseCategoricalCrossentropy`:
#This loss is equal to the negative log probability of the true class: 
#The loss is zero if the model is sure of the correct class.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# In[ ]:


loss_fn(y_train[:1], predictions).numpy()


# In[ ]:


#Before training, configuring and compiling the model
#Setting the optimizer class to adam, the loss to the loss_fn function defined earlier

model_2.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

#or as before but with sgd: model_2.compile(loss="sparse_categorical_crossentropy",
             #optimizer="sgd",
            # metrics=["accuracy"]) we use sparse... bc sparse labels and classes are exclusive


# In[ ]:


#training and evaluation
history=model_2.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

#or Model.evaluate` method checks the model's performance, usually on a validation set and test set
#model_2.evaluate(x_test,  y_test, verbose=2)


# In[ ]:


#model to return a probability, you can wrap the trained model, and attach the softmax to it:
probability_model = tf.keras.Sequential([model_2, tf.keras.layers.Softmax()])
probability_model(X_test[:5])


# ## TRYOUTS WTH SUBSET OF LENGTH 100

# In[7]:


#try-outs with smaller subsets:
X_train_subset = X_train[:100]
y_train_subset = y_train[:100]
X_test_subset = X_test[:100]
y_test_subset = y_test[:100]


# In[8]:


X_train_subset=X_train_subset/255.0
X_test_subset=X_test_subset/255.0


# In[9]:


X_train_subset_array = X_train_subset.to_numpy()  # Convert DataFrame to NumPy array

# Reshape the array
X_train_reshaped = X_train_subset_array.reshape(100,227, 227)
X_train_reshaped


# In[11]:


y_train_subset.dtype


# In[12]:


# Convert X_train_reshaped to float32
X_train_reshaped = np.array(X_train_reshaped, dtype=np.float32)
y_train_subset=np.array(y_train_subset, dtype=np.float32)

model_1.fit(X_train_reshaped, y_train_subset, epochs=10)


# In[15]:


X_test_subset_array = X_test_subset.to_numpy()
X_test_reshaped = X_test_subset_array.reshape(100,227, 227)
X_test_reshaped = np.array(X_test_reshaped, dtype=np.float32)
y_test_subset=np.array(y_test_subset, dtype=np.float32)


# In[18]:


test_loss, test_acc = model_1.evaluate(X_test_reshaped,  y_test_subset, verbose=2)

print('\nTest accuracy:', test_acc)


# In[20]:


#predictions
probability_model = tf.keras.Sequential([model_1, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(X_test_reshaped)
print(predictions[0])
#A prediction is an array of 10 numbers. They represent the model's "confidence" that the image corresponds to each of the 27 different logos.
#You can see which label has the highest confidence value by calling:
print(f'most probable logo is',np.argmax(predictions[0]))


# In[21]:


#the model is most confident that this image is 4
#Examining the test label shows that this classification is correct?False?:

y_test_subset[0]


# In[ ]:




