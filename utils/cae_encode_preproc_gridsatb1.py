#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This script provide functions to build and to train convolutional 
autoencoder for pre-processed NOAA-GridSat-B1 dataset.

The autoencoder is developed and teset with Tensorflow(2.5.0), 
CUDA 11.1/11.3 with cuDNN 8.1.0/8.2.0.
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
NY = 256
NX = 256
NC = 1

# To fix cublas not initialized problem
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.compat.v1.Session(config=config)

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
            yield (X,X) # a tuple with two numpy arrays with batch_size samples     
            batch_start += batch_size   
            batch_end += batch_size

def encode_preprocessed_noaagridsatb1(model, flist, batch_size):
    ''' Encode the data with pre-trained-model. '''
    nSample = len(flist)
    # Prepare for results
    results =[]
    # Read in and encode data by batch
    batch_start = 0
    batch_end = batch_size
    while batch_start < nSample:
        limit = min(batch_end, nSample)
        logging.info('Encoding batch: '+str(batch_start)+' - '+str(limit)+' of '+str(nSample)+'.')
        X = read_multiple_preprocessed_noaagridsatb1(flist['xuri'].iloc[batch_start:limit])
        encoded = model.encoder(X)
        # Flatten the encoded data
        tmp = [v.numpy().flatten() for v in encoded]
        results += tmp
        # Increment the loop   
        batch_start += batch_size
        batch_end += batch_size
    # Return results as an numpy array
    return(np.vstack(results))


#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Building convolutional autoencoder .')
    parser.add_argument('--datapath', '-i', help='the directory containing NOAA-GridSat-B1 data in numpy array.')
    parser.add_argument('--model', '-m', help='the pre-trained model directory.')
    parser.add_argument('--output', '-o', help='the prefix of output files.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    parser.add_argument('--batch_size', '-b', default=64, type=int, help='the batch size.')

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
    # Load model
    logging.info("Load pre-trained model from " + str(args.model))
    cae = tf.keras.models.load_model(args.model)
    logging.info(cae.encoder.summary())
    # Debug info
    nSample = datainfo.shape[0]
    logging.info("Ecnoding total data size: "+str(nSample))
    # Encode data with the autoencoder
    encoded = encode_preprocessed_noaagridsatb1(cae, datainfo, args.batch_size)
    # Prepare output
    pd.DataFrame(encoded, index=datainfo['timestamp']).to_csv(args.output+'_encoded.csv')
    np.save(args.output+'_encoded.npy', encoded)
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()
