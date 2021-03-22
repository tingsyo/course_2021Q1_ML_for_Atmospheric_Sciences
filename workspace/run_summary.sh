#!/bin/bash
for y in {2009..2009}
    do
        echo $y
        python ../utils/data_summarization_noaagridsatb1_by_image.py -i ../data/noaa/$y -o noaa_$y -l noaa_by_image_$y.log &
        python ../utils/data_summarization_noaagridsatb1_by_grid.py -i ../data/noaa/$y -o noaa_$y -b 128 -l noaa_by_grid_$y.log
    done 
