#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import keras
from keras.layers import Lambda, Input, Dense
from keras.models import Model,load_model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import tensorflow as tf
from transform_LC import transform_ztf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from keras import regularizers
import joblib
from keras.utils import plot_model
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
np.random.seed(42)

max_cores = 4

tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=max_cores,inter_op_parallelism_threads=max_cores)))
data_fname1 = '/home/sreejith/Downloads/snad/git/zwad/data/zr400/all1'
data_fname2 = '/home/sreejith/Downloads/snad/git/zwad/data/zr400/all2'

data1 = np.load(data_fname1 + '.npy',allow_pickle=True)
data2 = np.load(data_fname2 + '.npy',allow_pickle=True)

data1_new = np.pad(data1,((0,0),(67,67),(0,0)),'constant')
data2_new = np.pad(data2,((0,0),(0,1),(0,0)),'constant')

data = np.vstack((data1_new,data2_new))   


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


#train_data = np.expand_dims(data1,axis=-1)  

x_train,x_test,x_valid,x_ground= train_test_split(data,data,test_size=0.1,random_state=42)


original_dim = data.shape[1]*data.shape[2]

x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_valid = np.reshape(x_valid, [-1, original_dim])
x_ground = np.reshape(x_ground, [-1, original_dim])




# network parameters
input_shape = (original_dim, )
intermediate_dim = 801
batch_size = 100
latent_dim = 3
epochs = 100

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu',activity_regularizer=regularizers.l1(10e-5))(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_ztf_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)



# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')


# train the autoencoder
vae.compile(loss='binary_crossentropy',optimizer = RMSprop())

autoencoder_train = vae.fit(x_train,x_valid,shuffle=True,epochs=epochs,batch_size=batch_size,
                validation_data=(x_test, x_ground))


vae.save_weights('vae_mlp_ztf.h5')

loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


pred_train = vae.predict(x_train)
#latent_inputs


indx = np.random.choice(np.arange(0, pred_train.shape[0]))

plt.figure()
plt.plot(x_train[indx])
plt.plot(pred_train[indx])
plt.show()


