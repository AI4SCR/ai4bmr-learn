#!/bin/bash

#infile="/users/amarti51/prometex/data/H_E_Images/2024-05-24_n.1 - n.14/malexan3-24-05-2024-001.czi"
#basename=$(basename "$infile" .czi)
#bioformats2raw "$infile" "/users/amarti51/prometex/data/H_E_Images/2024-05-24_n.1 - n.14/${basename}_raw/"
#raw2ometiff "/users/amarti51/prometex/data/H_E_Images/2024-05-24_n.1 - n.14/${basename}_raw/" "/users/amarti51/prometex/data/H_E_Images/2024-05-24_n.1 - n.14/${basename}.ome.tiff" --pyramid
#vips copy "/users/amarti51/prometex/data/H_E_Images/2024-05-24_n.1 - n.14/${basename}.ome.tiff[page=0]" "/users/amarti51/prometex/data/H_E_Images/2024-05-24_n.1 - n.14/${basename}_openslide.tiff[tile,pyramid,compression=jpeg,bigtiff]"

# wget https://downloads.openmicroscopy.org/bio-formats/8.2.0/artifacts/bftools.zip -O bftools.zip
# cd /work/FAC/FBM/DBC/mrapsoma/prometex/data/beat/03_utils
# bfconvert -version || exit

input="/users/amarti51/prometex/data/beat/01_raw/H_E_Images/2024-05-24_n.1 - n.14/malexan3-24-05-2024-001.czi"
output="/users/amarti51/prometex/data/beat/01_raw/H_E_Images/2024-05-24_n.1 - n.14/malexan3-24-05-2024-001.ome.tiff"
bfconvert -nogroup -bigtiff -compression LZW -tilex 512 -tiley 512 "$input" "$output"

