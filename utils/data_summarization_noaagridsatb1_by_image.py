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
def summarize_single_image(img):
    ''' Calculate basic statistics of one Himawari-8 image. '''
    mean = np.mean(img.flatten())
    std = np.std(img.flatten())
    pt = np.percentile(img.flatten(), [0, 25, 50, 75,100])
    return({'mean':mean, 'stdev':std, 'min':pt[0],'pt25':pt[1],'median':pt[2],'pt75':pt[3], 'max':pt[4]})

def statistics_by_image(datainfo):
    ''' Given the data information, derive the statistics by image. '''
    list_stats = []
    for i in range(datainfo.shape[0]):
        row = datainfo.iloc[i,:]
        logging.debug(row['xuri'])
        tmp = read_noaagridsatb1(row['xuri'])
        stats = summarize_single_image(tmp)
        stats['timestamp'] = row['timestamp']
        list_stats.append(stats)
    return(pd.DataFrame(list_stats).sort_values('timestamp').reset_index(drop=True))

#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Summarize NOAA-GridSat-B1 by each image.')
    parser.add_argument('--datapath', '-i', help='the directory containing NOAA-GridSat-B1 data in netccdf format.')
    parser.add_argument('--output', '-o', help='the prefix of output files.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
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
    datainfo.to_csv(args.output+'.file_info.csv', index=False)
    # Derive per-image statistics
    logging.info('Deriving statistics per image.')
    stats_by_image = statistics_by_image(datainfo)
    stats_by_image.to_csv(args.output+'.stats_by_image.csv', index=False)
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()

