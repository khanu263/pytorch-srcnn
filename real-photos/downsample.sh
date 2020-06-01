#!/bin/bash

# Downsample real-world photos for testing, using
# factors 2-8. Inputs are 1680x1680.

for i in {0..7}
do
    convert original/$i.bmp -resize 840x840 z2/${i}_in.bmp
    convert original/$i.bmp -resize 560x560 z3/${i}_in.bmp
    convert original/$i.bmp -resize 420x420 z4/${i}_in.bmp
    convert original/$i.bmp -resize 336x336 z5/${i}_in.bmp
    convert original/$i.bmp -resize 280x280 z6/${i}_in.bmp
    convert original/$i.bmp -resize 240x240 z7/${i}_in.bmp
    convert original/$i.bmp -resize 210x210 z8/${i}_in.bmp
done
