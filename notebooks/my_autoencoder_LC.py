#!/usr/bin/env python
# coding: utf-8

# # Autoencoder 
# This was taken from https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial
# This is useless because for some reason the GPU stopped working

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #model will be trained on GPU 1
# In[1]:


# set max number of cores
import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
#get_ipython().run_line_magic('matplotlib', 'inline')
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,ZeroPadding2D, Cropping2D
from keras.models import Model
from keras.optimizers import RMSprop
import tensorflow as tf
from transform_LC import transform_ztf
import os
import pandas as pd

np.random.seed(42)

max_cores = 6

tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=max_cores, 
                                                                                            inter_op_parallelism_threads=max_cores)))


# transform ZTF data
# only do this once

data_fname1 = '/home/sreejith/Downloads/snad/git/zwad/data/zr400/all1'
data_fname2 = '/home/sreejith/Downloads/snad/git/zwad/data/zr400/all2'


if not os.path.isfile(data_fname1 + '.npy'):
    failed = transform_ztf(oids_file='/home/sreejith/Downloads/snad/git/zwad/data/zr400/zr400.txt', output_fname=data_fname1)
  
if not os.path.isfile(data_fname2 + '.npy'):
    failed = transform_ztf(oids_file='/home/sreejith/Downloads/snad/git/zwad/data/zr400/zr533.txt', output_fname=data_fname2)

data1 = np.load(data_fname1 + '.npy',allow_pickle=True)
data2 = np.load(data_fname2 + '.npy',allow_pickle=True)

data1.shape
data2.shape

data1_new = np.pad(data1,((0,0),(67,67),(0,0)),'constant')
data2_new = np.pad(data2,((0,0),(0,1),(0,0)),'constant')

data = np.vstack((data1_new,data2_new))   


'''
#indx = np.random.choice(np.arange(0, train_data.shape[0]))

plt.figure(figsize=[7,5])
plt.title(str(indx))
# Display the first image in training data
plt.errorbar(train_data[indx][:,0], train_data[indx][:,1], yerr=train_data[indx][:,2], fmt='o')
plt.show()


# ### Warning
# The tutorial says that data should be normalized.
# In our case this would lose meaning so it is not done here.

# ## Autoencoder

# In[17]:
# In[21]:
'''

batch_size = 200
epochs = 50
inChannel = 1
x, y = 534, 3

input_img = Input(shape = (x,y, inChannel))

input_img_padding = ZeroPadding2D((2,2))(input_img)



def autoencoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img_padding) #800 x 3 x 32
    pool1 = MaxPooling2D(pool_size=(2, 1),padding='same')(conv1) #400 x 3 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #400 x 3 x 64
    pool2 = MaxPooling2D(pool_size=(2, 1),padding='same')(conv2) #200 x 3 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #200 x 3 x 128 (small and thick)
    encoded = MaxPooling2D(pool_size=(2, 1),padding='same')(conv3)

    #decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #200 x 3 x 128
    up1 = UpSampling2D((2,1))(conv4) # 400 x 3 x 128
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 400 x 3 x 64
    up2 = UpSampling2D((2,1))(conv5) # 800 x 3 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='valid')(up2) # 800 x 3 x 1
    decoded_cropping = Cropping2D((2,1))(decoded)
    return decoded_cropping



autoencoder = Model(input_img, autoencoder(input_img))



autoencoder.compile(loss='mean_squared_error', optimizer = 'adam')


autoencoder.summary()



from sklearn.model_selection import train_test_split

train_data = np.expand_dims(data,axis=-1)  


train_X,valid_X,train_ground,valid_ground = train_test_split(train_data,
                                                             train_data, 
                                                             test_size=0.1, 
                                                          random_state=42)

# train the model
autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))





loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# ## Prediction on test data



#test_data = np.expand_dims(train_data,axis=-1)

pred_train = autoencoder.predict(train_data)



indx = np.random.choice(np.arange(0, pred_train.shape[0]))

x0 = 0

x = [x0]
y = [pred_train[indx][0][1][0]]
yerr = [pred_train[indx][0][2][0]]
                                                                                

xtrain0 = 0
xtrain = [xtrain0]
ytrain = [train_data[indx][0][1][0]]
yerrtrain = [train_data[indx][0][2][0]]

for i in range(1, pred_train.shape[1]):    
    # fix time axis for MJD
    x0 = x0 + pred_train[indx][i][0][0]
    x.append(x0)
    y.append(pred_train[indx][i][1][0])
    yerr.append(pred_train[indx][i][2][0])
    
    xtrain0 = xtrain0 + train_data[indx][i][0][0]
    xtrain.append(xtrain0)
    ytrain.append(train_data[indx][i][1][0])
    yerrtrain.append(train_data[indx][i][2][0])

plt.figure()
plt.errorbar(x, y, yerr=yerr, fmt='o', color='b', label='reconstructed')
plt.errorbar(xtrain, ytrain, yerr=yerrtrain, fmt='o', color='r', label='original')
plt.legend()
plt.show()





