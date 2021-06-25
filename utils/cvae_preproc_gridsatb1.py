#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This script provide functions to build and to train variational 
autoencoder for pre-processed NOAA-GridSat-B1 dataset.

The autoencoder is developed and teset with Tensorflow(2.5.0), 
CUDA 11.1/11.3 with cuDNN 8.1.0/8.2.0.
'''
import numpy as np
import pandas as pd
import os, argparse, logging, csv, h5py
import tensorflow as tf

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2020~2022, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2021-06-15'

# Parameters
NY = 256
NX = 256
NC = 1

# Utility functions
def list_preprocessed_gridsatb1_files(dir, suffix='.npy', to_remove=['.npy']):
    ''' To scan through the sapecified dir and get the corresponding file with suffix. '''
    import os
    import pandas as pd
    xfiles = []
    for root, dirs, files in os.walk(dir, followlinks=True):  # Loop through the directory
        for fn in files:
            if fn.endswith(suffix):         # Filter files with suffix
                timestamp = fn
                for s in to_remove:         # Removing prefix and suffix to get time-stamp
                    timestamp = timestamp.replace(s,'')
                xfiles.append({'timestamp':timestamp, 'xuri':os.path.join(root, fn)})
    return(pd.DataFrame(xfiles).sort_values('timestamp').reset_index(drop=True))

# Binary reader
def read_preprocessed_gridsatb1(furi):
    import numpy as np
    return(np.load(furi))

def read_multiple_preprocessed_noaagridsatb1(flist, flatten=False):
    ''' This method reads in a list of NOAA-GridSat-B1 images and returns a numpy array. '''
    import numpy as np
    data = []
    for f in flist:
        tmp = np.load(f)
        if flatten:
            tmp = tmp.flatten()
        data.append(np.expand_dims(tmp,-1))
    return(np.array(data))

def data_generator_ae(flist, batch_size, rseed=0):
    ''' Data generator for batched processing. '''
    nSample = len(flist)
    # Shuffle file list if specified
    if rseed!=0:
        flist = flist.sample(frac=1, random_state=rseed).reset_index(drop=True)
        logging.info('Shuffling the input data for batch processing with random seed: '+str(rseed))
    # This line is just to make the generator infinite, keras needs that    
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < nSample:
            limit = min(batch_end, nSample)
            X = read_multiple_preprocessed_noaagridsatb1(flist['xuri'].iloc[batch_start:limit])
            #print(X.shape)
            yield X # The customized optimization method takes only the input data
            batch_start += batch_size   
            batch_end += batch_size


# Define the customized layer
class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    
# Define the data dimension
latent_dim = 2048
inputx = 256
inputy = 256

# Define the encoder
def build_encoder(inputx, inputy, latent_dim):
    # Input
    encoder_inputs = tf.keras.Input(shape=(inputx, inputy, 1), name='encoder_input')
    # Reduce the input dimension with convolutional layers
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same", name='conv1')(encoder_inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same", name='conv2')(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", strides=2, padding="same", name='conv3')(x)
    # Map the convolutional kernels to Gaussian distributions
    x = tf.keras.layers.Flatten(name='flatten')(x)
    x = tf.keras.layers.Dense(256, activation="relu", name='dense')(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    # Define the encoder model
    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return(encoder)

# Define the decoder
def build_decoder(latent_dim):
    # Input from the latent space
    latent_inputs = tf.keras.Input(shape=(latent_dim,), name='decoder_input')
    # Map the latent vector into convolutional space
    x = tf.keras.layers.Dense(32 * 32 * 128, activation="relu")(latent_inputs)
    x = tf.keras.layers.Reshape((32, 32, 128))(x)
    # Increase the dimension with Conv2DTranspose
    x = tf.keras.layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same", name='convtr_1')(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same", name='convtr_2')(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same", name='convtr_3')(x)
    # Output to the original space
    decoder_outputs = tf.keras.layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same", name='convtr_4')(x)
    # Define the decoder model
    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    return(decoder)

# Define the autoencoder
class CVAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.mse_loss_tracker = tf.keras.metrics.Mean(name="mse_loss")
        self.sim_loss_tracker = tf.keras.metrics.Mean(name="sim_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.mse_loss_tracker,
            self.sim_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.metrics.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            mse_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.metrics.mean_squared_error(data, reconstruction), axis=(1, 2)
                )
            )
            sim_loss = tf.reduce_mean(tf.keras.losses.cosine_similarity(data, reconstruction))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss*0.0001 + kl_loss + mse_loss*0.01 + sim_loss*10
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.mse_loss_tracker.update_state(mse_loss)
        self.sim_loss_tracker.update_state(sim_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "mse_loss": self.mse_loss_tracker.result(),
            "sim_loss": self.sim_loss_tracker.result(),
        }
    
    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        decoded = self.decoder(z)
        return(decoded)

#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Building convolutional autoencoder .')
    parser.add_argument('--datapath', '-i', help='the directory containing NOAA-GridSat-B1 data in netCDF4 format.')
    parser.add_argument('--output', '-o', help='the prefix of output files.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    parser.add_argument('--batch_size', '-b', default=64, type=int, help='the batch size.')
    parser.add_argument('--random_seed', '-r', default=0, type=int, help='the random seed for shuffling.')
    parser.add_argument('--epochs', '-e', default=1, type=int, help='number of epochs.')
    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)
    # Get data files
    logging.info('Scanning data files.')
    datainfo = list_preprocessed_gridsatb1_files(args.datapath)
    # Define the data dimension
    latent_dim = 2048
    inputx = 256
    inputy = 256
    # Initialize the autoencoder
    logging.info("Building convolutional autoencoder with batch size of " + str(args.batch_size))
    encoder = build_encoder(inputx, inputy, latent_dim)
    decoder = build_decoder(latent_dim)
    cvae = CVAE(encoder, decoder)
    cvae.compile(optimizer=tf.keras.optimizers.Adam())
    # Debug info
    nSample = datainfo.shape[0]
    logging.info("Training autoencoder with data size: "+str(nSample))
    steps_train = np.ceil(nSample/args.batch_size)
    logging.info("Training data steps: " + str(steps_train))
    # Train the autoencoder
    hist = cvae.fit(data_generator_ae(datainfo, args.batch_size), 
        steps_per_epoch=steps_train,
        epochs=args.epochs)
    logging.info(cvae.summary())
    logging.info(cvae.encoder.summary())
    logging.info(cvae.decoder.summary())
    # Prepare output
    pd.DataFrame(hist.history).to_csv(args.output+'_hist.csv')
    cvae.save(args.output+'_model')
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()

