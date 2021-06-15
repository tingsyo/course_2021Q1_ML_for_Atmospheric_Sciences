#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This script provide functions that read the preprocessed NOAA-GridSat-B1 
dataset and transform with the pre-trained PCA model.

The PCA model is pre-trained with IncrementalPCA from sklearn, and stored with joblib.
'''
import numpy as np
import pandas as pd
import os, argparse, logging
from sklearn.decomposition import PCA, IncrementalPCA
import joblib, csv

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2019~2022, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2021-06-08'


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


# Incremental PCA
def transform_ipca_batch(ipca, finfo, batch_size=1024):
    ''' Use pretrained PCA to transform the data batch. '''   
    # Loop through finfo
    nSample = len(finfo)
    batch_start = 0
    batch_end = batch_size
    batch_count = 0
    # Process the first batch
    proj = None
    #
    while batch_start < nSample:
        logging.debug('Starting batch: '+str(batch_count))
        # Check bound
        limit = min(batch_end, nSample)
        # Read batch data
        data = read_multiple_prerpocessed_noaagridsatb1(finfo['xuri'].iloc[batch_start:limit], flatten=True)
        logging.debug(data.shape)
        # increment
        batch_start = limit   
        batch_end = limit + batch_size
        batch_count += 1
        # Partial fit with batch data
        if proj is None:
            proj = ipca.transform(data)
        else:
            proj = np.vstack((proj,ipca.transform(data)))
    return(proj)

#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Performing Incremental PCA on NOAA-GridSat-B1 data.')
    parser.add_argument('--datapath', '-i', help='the directory containing preprocessed NOAA-GridSat-B1 data in npy format.')
    parser.add_argument('--output', '-o', help='the prefix of output files.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    parser.add_argument('--model_file', '-m', help='the joblib file storing the pre-trained model.')
    parser.add_argument('--batch_size', '-b', default=1024, type=int, help='the batch size.')
    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)
    # Get data files
    datainfo = list_prerpocessed_gridsatb1_files(args.datapath)
    logging.info('Scanning data files: '+str(datainfo.shape[0]))
    # Load Pre-trained PCA Model
    ipca = joblib.load(args.model_file)
    logging.info("Loading pre-trained PCA model from: "+ str(args.model_file)+", model dimension: "+str(ipca.components_.shape))
    # Preparing output
    proj = transform_ipca_batch(ipca, datainfo, batch_size=args.batch_size)
    projections = pd.DataFrame(proj)
    projections.index = datainfo['timestamp']
    logging.info("Writing transformed data to "+args.output+", data dimension: "+str(projections.shape))
    projections.to_csv(args.output)
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()

