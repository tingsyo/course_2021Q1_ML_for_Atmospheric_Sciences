#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This script provide functions to build and to train convolutional 
autoencoder for pre-processed NOAA-GridSat-B1 dataset.

The autoencoder is developed with Tensorflow(2.4.1).
'''
import numpy as np
import pandas as pd
import os, argparse, logging, csv, h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose


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
NY = 858
NX = 858
NC = 1

# Utility functions
def list_prerpocessed_gridsatb1_files(dir, suffix='.npy', to_remove=['.npy']):
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
def read_prerpocessed_gridsatb1(furi):
    import numpy as np
    return(np.load(furi))

def read_multiple_prerpocessed_noaagridsatb1(flist, flatten=False):
    ''' This method reads in a list of NOAA-GridSat-B1 images and returns a numpy array. '''
    import numpy as np
    data = []
    for f in flist:
        tmp = np.load(f)
        if flatten:
            tmp = tmp.flatten()
        data.append(tmp)
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
            X = read_multiple_prerpocessed_noaagridsatb1(flist['xuri'].iloc[batch_start:limit])
            #print(X.shape)
            yield (X,X) # a tuple with two numpy arrays with batch_size samples     
            batch_start += batch_size   
            batch_end += batch_size

class CAE(tf.keras.Model):
    """Convolutional autoencoder."""

    def __init__(self, inputx, inputy):
        super(CAE, self).__init__()
        self.inputx = inputx
        self.inputy = inputy
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(inputx, inputy, 1), name='encoder_input'),
                tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), padding='same',
                    activation='relu', name='encoder_conv1'),
                tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(2, 2), padding='same', 
                    activation='relu', name='encoder_conv2'),
                tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(2, 2), padding='same', 
                    activation='relu', name='encoder_conv3'),
                tf.keras.layers.Conv2D(filters=2, kernel_size=3, strides=(1, 1), padding='same', 
                    activation='relu', name='encoder_conv4'),
            ], name='encoder'
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(32,32,2)),
                tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=3, strides=2, padding='same',
                    activation='relu', name='decoder_conv1'),
                tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same',
                    activation='relu', name='decoder_conv2'),
                tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same',
                    activation='relu', name='decoder_conv3'),
                # No activation
                tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same', 
                    activation='sigmoid', name='decoder_outoput'),
            ], name='decoder'
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


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
    datainfo = list_prerpocessed_gridsatb1_files(args.datapath)
    # Initialize the autoencoder
    logging.info("Building convolutional autoencoder with batch size of " + str(args.batch_size))
    cae = CAE(inputx=256, inputy=256)
    # Debug info
    nSample = datainfo.shape[0]
    logging.info(ae[0].summary())
    logging.info("Training autoencoder with data size: "+str(nSample))
    steps_train = np.ceil(nSample/args.batch_size)
    logging.info("Training data steps: " + str(steps_train))
    # Train the autoencoder
    hist = ae[0].fit(data_generator_ae(datainfo, args.batch_size), 
        steps_per_epoch=steps_train,
        epochs=args.epochs)
    # Prepare output
    pd.DataFrame(hist.history).to_csv(args.output+'_hist.csv')
    cae.save(args.output+'_cae.h5')
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()

