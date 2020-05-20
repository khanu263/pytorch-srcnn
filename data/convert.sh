#!/bin/bash

# Convert files in given folder to BMP3 format,
# and optionally rename in number sequence.

# Set variables
dir=$1
files=$dir"*"
i=0

# Iterate through files
for f in $files
do
    if [ $2 -eq 1 ]
    then
        new=$(printf "%02d.bmp" "$i")
        convert $f BMP3:$dir$new
        rm $f
        let i=i+1
    else
        convert $f BMP3:$f
    fi
done

