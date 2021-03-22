#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This script provide functions that build convolutional autoencoder to NOAA-GridSat-B1 dataset.
The domain of the raw data ranged from -70' to 69.93'N, -180' to 179.94'E, with 0.07' intervals. 
The data dimension is (1, 2000, 5143), and missing values is -31999.

According to the official how to, the variable(irwin_cdr) contains int16 with values:
    Value = Integer * scale_factor + add_offset
, where scale_factor = 0.01 and offset = 200 (Earth Engine Data Catalog ).

To focus our analysis to East Asia, we cut off a square of 0'~60'N, 100'~160'E (858,858).

The autoencoder is developed with Tensorflow(2.3.1).
'''
import numpy as np
import pandas as pd
import os, argparse, logging, csv, h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2020~2022, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2020-10-01'

# Parameters
NY = 858
NX = 858
NC = 1

# Utility functions
def list_noaagridsatb1_files(dir, suffix='.v02r01.nc', to_remove=['GRIDSAT-B1.','.v02r01.nc']):
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
def read_noaagridsatb1(furi, var='irwin_cdr', scale=0.01, offset=200, remove_na=True, crop_east_asia=True):
    ''' The method reads in a NOAA-GridSta-B1 image in netCDF4 format (.nc file). 
        The brightness temperature data was stored in int16 as 'irwin_cdr', with 
        a scal factor of 0.01 and offset of 200. The missing values is flagged as -31999.
        More details of the data is described in https://developers.google.com/earth-engine/datasets/catalog/NOAA_CDR_GRIDSAT-B1_V2.
        Since our analysis focuss on East Asia (0-60'N, 100-160'E), we used an 
        option to crop the data to this region (index: lat:1000~1858, lon:4000~4858).
        The output is a 2-d numpy array of float32 with shape (858, 858).
    '''
    import numpy as np
    import netCDF4 as nc
    # Read in data
    data = nc.Dataset(furi)
    cdr = np.array(data.variables['irwin_cdr'])*scale+offset
    # Remove missing value
    if remove_na:
        cdr[cdr<0] = offset
    # Crop domain to East-Asia (0-60'N, 100-160'E)
    if crop_east_asia:
        return(cdr[0, 1000:1858, 4000:4858])
    else:
        return(cdr[0,:,:])

def read_multiple_noaagridsatb1(flist, flatten=False):
    ''' This method reads in a list of NOAA-GridSat-B1 images and returns a numpy array. '''
    import numpy as np
    data = []
    for f in flist:
        tmp = read_noaagridsatb1(f)
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
            X = read_multiple_noaagridsatb1(flist['xuri'].iloc[batch_start:limit])
            #print(X.shape)
            yield (X,X) # a tuple with two numpy arrays with batch_size samples     
            batch_start += batch_size   
            batch_end += batch_size


def initialize_conv_autoencoder_noaagridsatb1(input_shape):
    ''' The convolutional autoencoder. '''
    # Define input layer
    input_data = Input(shape=input_shape)  # adapt this if using `channels_first` image data format
    # Define encoder layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='encoder_conv1')(input_data)
    x = MaxPooling2D((2, 2), name='encoder_maxpool1')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='encoder_conv2')(x)
    x = MaxPooling2D((3, 3), name='encoder_maxpool2')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='encoder_conv3')(x)
    encoded = MaxPooling2D((11, 11), name='encoder_maxpool3')(x)
    # Define decoder layers
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='decoder_conv1')(encoded)
    x = UpSampling2D((11, 11), name='decoder_upsamp1')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='decoder_conv2')(x)
    x = UpSampling2D((3, 3), name='decoder_upsamp2')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='decoder_conv3')(x)
    x = UpSampling2D((2, 2), name='decoder_upsamp4')(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', name='decoder_output', padding='same')(x)
    # Define autoencoder
    autoencoder = Model(input_data, decoded)
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['cosine_similarity'])
    # Encoder
    encoder = Model(input_data, encoded)
    return((autoencoder, encoder))

#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Building convolutional autoencoder .')
    parser.add_argument('--datapath', '-i', help='the directory containing NOAA-GridSat-B1 data in netCDF4 format.')
    parser.add_argument('--output', '-o', help='the prefix of output files.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    parser.add_argument('--batch_size', '-b', default=128, type=int, help='the batch size.')
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
    datainfo = list_noaagridsatb1_files(args.datapath)
    # Initialize the autoencoder
    logging.info("Building convolutional autoencoder with batch size of " + str(args.batch_size))
    ae = initialize_conv_autoencoder_noaagridsatb1((NY, NX, NC))
    # Debug info
    nSample = datainfo.shape[0]
    logging.info(ae[0].summary())
    logging.info("Training autoencoder with data size: "+str(nSample))
    steps_train = np.ceil(nSample/args.batch_size)
    logging.info("Training data steps: " + str(steps_train))
    # Train the autoencoder
    hist = ae[0].fit(data_generator_ae(datainfo, args.batch_size, rseed=args.random_seed), 
        steps_per_epoch=steps_train,
        epochs=args.epochs,
        max_queue_size=args.batch_size,
        verbose=1)
    # Prepare output
    pd.DataFrame(hist.history).to_csv(args.output+'_hist.csv')
    ae[0].save(args.output+'_ae.h5')
    ae[1].save(args.output+'_encoder.h5')
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()

