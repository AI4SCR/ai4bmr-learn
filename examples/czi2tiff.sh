#!/bin/bash

infile="myimage.czi"
basename=$(basename "$infile" .czi)

bioformats2raw "$infile" "${basename}_raw/"
raw2ometiff "${basename}_raw/" "${basename}.ome.tiff" --pyramid
vips copy "${basename}.ome.tiff[page=0]" "${basename}_openslide.tiff[tile,pyramid,compression=jpeg,bigtiff]"