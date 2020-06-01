#!/bin/bash

# Run upscaling on all photos at all zoom levels.
# (Things are hard-coded, probably not useful.)

# Move to root directory
cd ..

# Run through photos
for i in {2..8}
do
    for j in {0..7}
    do
        python upscale.py -z $i -e 7000 -i real-photos/z$i/$j.bmp -o real-photos/z$i/${j}_bicubic.bmp -s real-photos/original/$j.bmp -b
        python upscale.py -z $i -e 7000 -i real-photos/z$i/$j.bmp -o real-photos/z$i/${j}_srcnn.bmp -s real-photos/original/$j.bmp
    done
done
