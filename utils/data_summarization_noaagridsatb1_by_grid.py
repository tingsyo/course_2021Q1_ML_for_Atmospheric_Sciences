#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
The first step of analysis is to explore the basic statistics of the Himawari-8 images.
The dimension of our dataset:
  - 3300 x 3300 pixels
  - 15' ~ 30'N / 114' ~ 129' E (~500m in real space)
  - And the data came in every 10 minutes if there is no missing.
We will scan through the dataset, sumarize the data values:
1. image by image
2. grid by grid
'''

import numpy as np
import pandas as pd
import os, argparse, logging

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
    for root, dirs, files in os.walk(dir):  # Loop through the directory
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

def read_multiple_noaagridsatb1(flist):
    ''' This method reads in a list of NOAA-GridSat-B1 images and returns a numpy array. '''
    import numpy as np
    data = []
    for f in flist:
        data.append(read_noaagridsatb1(f))
    return(np.array(data))

# Statistical summary
# Per-grid analysis
def summarize_by_grid(data):
    ''' Given a 3D numpy array, calculate the mean and variance on the first axis. '''
    mean_image = np.mean(data, axis=0)
    var_image = np.var(data, axis=0)
    return({'mean':mean_image, 'variance':var_image})

def summarize_noaagridsatb1_by_grid(flist, batch_size=None, shuffle=True, rseed=123):
    ''' Calculate grid-by-grid statistics of a list of NOAA-GridSat-B1 images. '''
    if batch_size is None:  # Read in all data
        data = read_multiple_noaagridsatb1(flist)
        tmp = summarize_by_grid(data)
        summary={'mean':tmp['mean'], 'stdev':np.sqrt(tmp['variance'])}
    else:                   # Read in data by batch
        # Shuffle flist for random batching
        if shuffle:
            flist = flist.sample(frac=1, random_state=rseed).reset_index(drop=True)
            logging.debug('Shuffling the input data for batch processing.')
        pooled_mean = None
        pooled_var = None
        nSample = len(flist)
        batch_start = 0
        batch_end = batch_size
        batch_count = 0
        # Loop through all files
        while batch_start < nSample:
            limit = min(batch_end, nSample)
            logging.debug("Batch "+str(batch_count)+', size:'+str(limit-batch_start))
            data = read_multiple_noaagridsatb1(flist[batch_start:limit])
            logging.debug(flist[batch_start])
            # calculate statistics by increment
            tmp = summarize_by_grid(data)
            if pooled_mean is None:
                pooled_mean = (limit - batch_start)*tmp['mean']
                pooled_var = (limit - batch_start - 1)*tmp['variance']
            else:
                pooled_mean += (limit - batch_start)*tmp['mean']
                pooled_var += (limit - batch_start - 1)*tmp['variance']
            # increment
            batch_start += batch_size   
            batch_end += batch_size
            batch_count += 1
        # Pooling
        pooled_mean = pooled_mean/nSample
        pooled_var = pooled_var/(nSample-batch_count)
        summary={'mean':pooled_mean, 'stdev':np.sqrt(pooled_var)}
    # 
    return(summary)


#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--datapath', '-i', help='the directory containing Himawari data in btp format.')
    parser.add_argument('--output', '-o', help='the prefix of output files.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    parser.add_argument('--batch_size', '-b', default=32, type=int, help='the batch size.')
    parser.add_argument('--random_seed', '-r', default=123, type=int, help='the random seed for shuffling.')
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
    # Derive per-grid statistics
    logging.info('Deriving statistics per grid with batch size:'+str(args.batch_size))
    stats_by_grid = summarize_noaagridsatb1_by_grid(datainfo['xuri'], batch_size=args.batch_size, rseed=args.random_seed)
    stats_by_grid['mean'].astype('float32').tofile(args.output+'_mean.npy')
    stats_by_grid['stdev'].astype('float32').tofile(args.output+'_std.npy')
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()

