#import
import numpy as np 
import pandas as pd 
import cv2 as cv
from PIL import Image
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPool2D, BatchNormalization, Activation, Dropout, 
    Flatten, Dense
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


# In[2]:


# Load the dataset 
data = np.load("/home/mohit/Desktop/IISERB/AI/dataset.npy", allow_pickle=True)
print("Shape of Data is:", data.shape) # Shape of data


# ### Explanation of the dataset
# #### 11446 x 2 represents a 11446 x 2 matrix with first entry as images and second as the class label of it.
# #### While creating the dataset we transformed our images from size from 512 x 512 x 3 to 228 x 228 x 3, 3 is the number of channels in the image since it is a RGB image.
# 

# In[3]:


# First image
img1 = data[0][0] 
plt.imshow(img1)


# In[4]:


image = Image.fromarray(img1)
img = image.resize((200, 200), Image.BILINEAR) 

plt.imshow(img)


# In[5]:


# Displaying one image
# Get the label and image data for the first image
label = data[0][1] 
img1 = data[0][0] 

# Show the image with its label as the x-axis label
plt.imshow(img1)
plt.xlabel(f"{label}")
plt.show()


# In[6]:


# Creating features and target variables 
X = [] # Features 
Y = [] # Target

# Iterate over each data point and check if its label is 1, 2, or 3
for x in range(data.shape[0]):
    if data[x][1] == 1 or data[x][1] == 2 or data[x][1] == 3:
        X.append(data[x][0])
        Y.append(data[x][1])

# Print the number of elements in X and Y
print(f"Number of elements in X are {len(X)} and in Y are {len(Y)}")


# In[7]:


# Count the number of occurrences of each label
label_counts = Counter(Y)

# Create a bar chart of the label counts
plt.bar(label_counts.keys(), label_counts.values(), width=0.5)
plt.show()
print("The above plot shows that there is a high class imbalance and we need to resolve it.")


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Set random state for reproducibility
np.random.RandomState(seed=42)

# Perform label encoding
y = np.array(Y).reshape(-1,1)
encoder = OneHotEncoder()
encoder.fit(y)
y_encoded = encoder.transform(y).toarray()

# Split the data into train, test, and validation sets
x_train_val, x_test, y_train_val, y_test = train_test_split(
    np.array(X), y_encoded, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_val, y_train_val, test_size=0.1, random_state=42)

# Print the shapes of the train, test, and validation sets
print("**********************")
print("Shape of X Train data is : ",x_train.shape,"\n**********************")
print("Shape of Y Train data is : ",y_train.shape,"\n**********************")
print("Shape of X test data is : ",x_test.shape,"\n**********************")
print("Shape of Y test data is : ",y_test.shape,"\n**********************")
print("Shape of X Validation data is : ",x_val.shape,"\n**********************")
print("Shape of Y Validation data is : ", y_val.shape,"\n**********************")


# In[9]:


# normalize pixel values 
x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

# print confirmation message
print("Pixel Values normalized")


# In[10]:


# calculating the class weights and then we will use them to fit into our model
class_weights = compute_class_weight('balanced', classes=np.unique(Y), y=Y)

# convert the class weights to a dictionary to pass to the model
class_weights_dict = dict(enumerate(class_weights))

# print the class weights dictionary
print(class_weights_dict)


# ### Applying a Convolutional Neural Network for the task of classification.

# In[11]:


# Define the CNN model
model = Sequential([
    Conv2D(filters=40, kernel_size=(3, 3), activation='relu', input_shape=(228, 228, 3)),
    BatchNormalization(),
    Conv2D(filters=40, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPool2D(strides=(2, 2)),
    Dropout(0.25),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPool2D(strides=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.25),
    Dense(780, activation='relu'),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(3, activation='softmax')
])

learning_rate = 0.001

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate),
    metrics=['accuracy']
)

# Print model summary
model.summary()


# In[12]:


#creating checkpoints to prevent model from overfitting by monitoring the validation accuracy
model_path="D:/SEM6/ArtificialIntelligence/Project"
save_best = ModelCheckpoint (model_path, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='max')
early_callback=tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=8,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True)
reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=5,
    verbose=0,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0)


# In[ ]:


tf.config.run_functions_eagerly(True)
history = model.fit( x_train, y_train, 
                    epochs = 50, batch_size = 100, 
                    callbacks=[save_best,early_callback,reduce_lr], verbose=1, 
                    validation_data = (x_val, y_val),class_weight=class_weights_dict)


# In[4]:


plt.figure(figsize=(6, 5))
plt.plot(history.history['accuracy'], color='r')
plt.plot(history.history['val_accuracy'], color='b')
plt.title('Model Accuracy', weight='bold', fontsize=16)
plt.ylabel('accuracy', weight='bold', fontsize=14)
plt.xlabel('epoch', weight='bold', fontsize=14)
plt.ylim(0.1, 0.9)
plt.xticks(weight='bold', fontsize=12)
plt.yticks(weight='bold', fontsize=12)
plt.legend(['train', 'val'], loc='upper left', prop={'size': 14})
plt.grid(color = 'y', linewidth='0.5')
plt.show()


# In[5]:


plt.figure(figsize=(6, 5))
plt.plot(history.history['loss'], color='r')
plt.plot(history.history['val_loss'], color='b')
plt.title('Model Loss', weight='bold', fontsize=16)
plt.ylabel('Loss', weight='bold', fontsize=14)
plt.xlabel('epoch', weight='bold', fontsize=14)
# plt.ylim(0.1, 0.9)
plt.xticks(weight='bold', fontsize=12)
plt.yticks(weight='bold', fontsize=12)
plt.legend(['train', 'val'], loc='upper left', prop={'size': 14})
plt.grid(color = 'y', linewidth='0.5')
plt.show()


# ### Evaluating the model

# In[6]:


model_eval = tf.keras.models.load_model('D:/SEM6/ArtificialIntelligence/Project')
score = model.evaluate(x_test, y_test, verbose=0)
print('Accuracy over the test set: \n ', round((score[1]*100), 2), '%')


# ### Feature Extraction

# In[7]:


feature_extractor = keras.Model(
   inputs=model.inputs,
   outputs=model.get_layer(name="conv2d_11").output,
)


# In[8]:


intermediate_output_test = feature_extractor(x_test)


# In[9]:


test=intermediate_output_test.numpy()


# In[10]:


#to remove memory error 

