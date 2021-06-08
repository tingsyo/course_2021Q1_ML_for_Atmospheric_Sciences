#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This script provide functions that read and perform pre-rpocessing on the NOAA-GridSat-B1 dataset.

    1. Read raw data: -70' to 69.93'N, -180' to 179.94'E, with 0.07' intervals. shape=(2000, 5143)
    2. Crop to East Asia region (100-160E, 0-60N) -> shape=(858,858) 
    3. Resize the cropped data into a domain of (2^N, 2^N) for better processing. (using opencv2)
      a. cv2.resize(512,512) -> (512,512)
      b. cv2.resize(256,256) -> (256,256)
    4. Rescale the values of white temperature to (0,1). (divided by the max value)
'''
import numpy as np
import pandas as pd
import os, argparse, logging, csv

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2019~2022, DataQualia Lab Co. Ltd."
__credits__ = ["Ting-Shuo Yo"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2021-06-05'


# Utility functions
def list_noaagridsatb1_files(dir, suffix='.00.v02r01.nc', to_remove=['GRIDSAT-B1.','.00.v02r01.nc']):
    ''' To scan through the sapecified dir and get the corresponding file with suffix. 
        In this script, we add '.00' to use one image per day (00z).
    '''

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

# Resize with opencv
def resize_INTER_CUBIC(data, out_size=512):
    '''This method resizes the data (2d-array with equal width-heght) into the shape (out_size, out_size).
       OpenCV(cv2) with cubic-spline interpolation is used.
    '''
    import cv2
    resized = cv2.resize(data, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
    return(resized)



#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Performing Incremental PCA on NOAA-GridSat-B1 data.')
    parser.add_argument('--datapath', '-i', help='the directory containing NOAA-GridSat-B1 data in netCDF4 format.')
    parser.add_argument('--outputpath', '-o', default='.', help='the directory for output files.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    parser.add_argument('--resize', '-w', default=512, type=int, help='the width-height of the output data.')
    parser.add_argument('--min', default=117, type=int, help='the scale factor of bright-temperature.')
    parser.add_argument('--max', default=354, type=int, help='the scale factor of bright-temperature.')

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
    # Perform pre-processing file by file
    logging.info("Performing preprosessing on NOAA-GridSat-B1 with resizing to "+ str(args.resize)
        +" and scaling by (" + str(args.min) + ", " + str(args.max) + ").")
    # Preparing output
    OUTDIR256 = args.outputpath + '/256'
    if not os.path.exists(OUTDIR256):
        os.makedirs(OUTDIR256)
    OUTDIR512 = args.outputpath + '/512'
    if not os.path.exists(OUTDIR512):
        os.makedirs(OUTDIR512)
    # 
    for i in range(datainfo.shape[0]):
        data = read_noaagridsatb1(datainfo['xuri'].iloc[i])
        d256 = (resize_INTER_CUBIC(data, out_size=256)-args.min)/(args.max-args.min)
        f256 = os.path.join(OUTDIR256, datainfo['timestamp'].iloc[i]+'.npy')
        np.save(f256, d256)
        d512 = (resize_INTER_CUBIC(data, out_size=512)-args.min)/(args.max-args.min)
        f512 = os.path.join(OUTDIR512, datainfo['timestamp'].iloc[i]+'.npy')
        np.save(f512, d512)
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()

