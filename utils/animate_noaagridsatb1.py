#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This script provide functions that read NOAA-GridSat-B1 dataset and create an animation. Due to
some problems between matplotlib.animation and Basemap, we need to output each image separately 
and then aggregate them into one movie file.
'''
import numpy as np
import pandas as pd
import os, argparse, logging
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import cv2

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

def create_noaagridsatb1_images(finfo, workspace):
    # Setting parameters
    lat0 = 0.0
    lat1 = 60.059998
    lon0 = 100.0
    lon1 = 160.06
    # Loop through file-list
    for i in range(finfo.shape[0]):
        data = read_noaagridsatb1(finfo['xuri'].iloc[i])
        m = Basemap(llcrnrlon=lon0, urcrnrlon=lon1, llcrnrlat=lat0, urcrnrlat=lat1, resolution='l')
        m.drawcoastlines()
        m.imshow(data, alpha=0.99, cmap='Greys', vmin=180, vmax=300)
        m.colorbar()
        plt.title(finfo['timestamp'].iloc[i])
        plt.savefig(workspace+'/'+finfo['timestamp'].iloc[i]+'.png')
        plt.close()
    return(0)

def images_to_mp4(finfo, workspace, output):
    # Determine the width and height from the first image
    image_path = workspace + '/' + finfo['timestamp'].iloc[0] + '.png'
    frame = cv2.imread(image_path)
    #cv2.imshow('video',frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 4.0, (width, height))

    # Loop through file-list
    for i in range(finfo.shape[0]):
        image_path = workspace + '/' + finfo['timestamp'].iloc[i] + '.png'
        logging.debug('Encoding '+image_path)
        frame = cv2.imread(image_path)
        out.write(frame)
        #cv2.imshow('video',frame)
        #if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        #    break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
    #
    return(0)  

def create_noaagridsatb1_movie(finfo, output, workspace='./', fps=32):
    # Setting parameters for Basemap
    lat0 = 0.0
    lat1 = 60.059998
    lon0 = 100.0
    lon1 = 160.06
    # Initialize movie file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, fps, (512, 512))
    # Loop through file-list
    for i in range(finfo.shape[0]):
        logging.debug('    Encoding '+finfo['timestamp'].iloc[i])
        # Load raw data
        data = read_noaagridsatb1(finfo['xuri'].iloc[i])
        # Create the frame
        plt.figure(figsize=(5.12, 5.12), dpi=100)
        m = Basemap(llcrnrlon=lon0, urcrnrlon=lon1, llcrnrlat=lat0, urcrnrlat=lat1, resolution='l')
        m.drawcoastlines()
        m.imshow(data, alpha=0.99, cmap='Greys', vmin=180, vmax=300)
        m.colorbar()
        plt.title(finfo['timestamp'].iloc[i])
        plt.savefig(workspace+'/temp.png')
        # Load image and write to frame
        frame = cv2.imread(workspace+'/temp.png')
        out.write(frame)
    #
    out.release()
    cv2.destroyAllWindows()    
    return(0)

#-----------------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--datapath', '-i', help='the directory containing Himawari data in btp format.')
    parser.add_argument('--output', '-o', help='the prefix of output files.')
    parser.add_argument('--workspace', '-w', default='./', help='temporary workspace for storing images.')
    parser.add_argument('--fps', '-f', default=32, help='Frame rate for the output movie.')
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
    #datainfo.to_csv(args.output+'.file_info.csv', index=False)
    # Convert data into image
    #logging.info("Converting dataset into images")
    #toimg = create_noaagridsatb1_images(datainfo, workspace=args.workspace)
    # Convert images into movie
    #logging.info("Converting dataset into images")
    #tomp4 = images_to_mp4(datainfo, workspace=args.workspace, output=args.output)
    logging.info('Creating movie.')
    create_noaagridsatb1_movie(datainfo, output=args.output, workspace=args.workspace, fps=float(args.fps))
    # done
    return(0)
    
#==========
# Script
#==========
if __name__=="__main__":
    main()

