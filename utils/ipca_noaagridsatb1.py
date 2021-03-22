#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This script provide functions that read and perform PCA with NOAA-GridSat-B1 dataset.
The domain of the raw data ranged from -70' to 69.93'N, -180' to 179.94'E, with 0.07' intervals. 
The data dimension is (1, 2000, 5143), and missing values is -31999.

According to the official how to, the variable(irwin_cdr) contains int16 with values:
    Value = Integer * scale_factor + add_offset
, where scale_factor = 0.01 and offset = 200 (Earth Engine Data Catalog ).

To focus our analysis to East Asia, we cut off a square of 0'~60'N, 100'~160'E (858,858).

The PCA is performed with IncrementalPCA from sklearn.
'''
import numpy as np
import pandas as pd
import os, argparse, logging
from sklearn.decomposition import PCA, IncrementalPCA
import joblib, csv

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2019~2020, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2020-07-20'


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


# Incremental PCA
def fit_ipca_partial(finfo, n_component=20, batch_size=128, rseed=0):
    ''' Initial and fit a PCA model with incremental PCA. '''
    # Shuffle file list if specified
    if rseed!=0:
        flist = flist.sample(frac=1, random_state=rseed).reset_index(drop=True)
        logging.info('Shuffling the input data for batch processing with random seed: '+str(rseed))
    # Initialize the PCA object
    ipca = IncrementalPCA(n_components=n_component, batch_size=batch_size)
    # Loop through finfo
    nSample = len(finfo)
    batch_start = 0
    batch_end = batch_size
    batch_count = 0
    while batch_start < nSample:
        logging.debug('Starting batch: '+str(batch_count))
        # Check bound
        limit = min(batch_end, nSample)         # Check for the final batch
        if n_component>(nSample-batch_end):     # Merge the final batch if it's too small
            logging.info('The final batch is too small, merge it to the previous batch.')
            limit = nSample
        # Read batch data
        data = read_multiple_noaagridsatb1(finfo['xuri'].iloc[batch_start:limit], flatten=True)
        logging.debug(data.shape)
        # increment
        batch_start = limit   
        batch_end = limit + batch_size
        batch_count += 1
        # Partial fit with batch data
        ipca.partial_fit(data)
    #
    return(ipca)


def writeToCsv(output, fname, header=None):
    # Overwrite the output file:
    with open(fname, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=csv.QUOTE_ALL)
        if header is not None:
            writer.writerow(header)
        for r in output:
            writer.writerow(r)
    return(0)
#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Performing Incremental PCA on NOAA-GridSat-B1 data.')
    parser.add_argument('--datapath', '-i', help='the directory containing NOAA-GridSat-B1 data in netCDF4 format.')
    parser.add_argument('--output', '-o', help='the prefix of output files.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    parser.add_argument('--n_component', '-n', default=50, type=int, help='the number of PCs to derive.')
    parser.add_argument('--batch_size', '-b', default=128, type=int, help='the batch size.')
    parser.add_argument('--random_seed', '-r', default=0, type=int, help='the random seed for shuffling.')
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
    #datainfo.to_csv(args.output+'.file_info.csv', index=False)
    # IncrementalPCA
    logging.info("Performing IncrementalPCA with "+ str(args.n_component)+" components and batch size of " + str(args.batch_size))
    ipca = fit_ipca_partial(datainfo, n_component=args.n_component, batch_size=args.batch_size, rseed=args.random_seed)
    # Preparing output
    ev = ipca.explained_variance_
    evr = ipca.explained_variance_ratio_
    com = np.transpose(ipca.components_)
    logging.info("Explained variance ratio: "+ str(evr))
    # Output components
    com_header = ['pc'+str(x+1) for x in range(args.n_component)]
    #writeToCsv(com, args.output+'.components.csv', header=com_header)
    pd.DataFrame({'ev':ev, 'evr':evr}).to_csv(args.output+'.exp_var.csv')
    # Output fitted IPCA model
    joblib.dump(ipca, args.output+".pca.mod")
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()

