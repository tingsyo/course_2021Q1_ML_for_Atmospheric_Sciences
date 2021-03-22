#!/bin/bash
PROJ_LIB=$HOME/anaconda3/share/proj
for y in {1999..2018}
    do
        echo $y
        #python ../utils/animate_noaagridsatb1_png.py -i ../data/noaa/$y -o noaa_$y.mp4 -f 32 -w ./noaa_png/ -l noaa_animation_$y.log 
        python ../utils/animate_noaagridsatb1.py -i ../data/noaa/$y -o noaa_$y.mp4 -f 32 -l noaa_animation_$y.log 
    done 
