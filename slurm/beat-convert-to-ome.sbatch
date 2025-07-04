#!/usr/bin/env bash
#SBATCH --job-name=proj=beat-task=convert-to-ome
#SBATCH --output=/users/amarti51/logs/proj=beat-task=convert-to-ome-%j.log
#SBATCH --error=/users/amarti51/logs/proj=beat-task=convert-to-ome-%j.err

#SBATCH --account=mrapsoma_prometex

#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

ROOT_DIR="/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/fmx/data/datasets/beat"
BFCONVERT="/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/beat/03_tools/bftools/bfconvert"

find "$ROOT_DIR" -type f -name "region.tif" | while read -r tif; do
  dir=$(dirname "$tif")
  base=$(basename "$tif" .tif)
  out="$dir/${base}.ome.tif"
  echo "Processing $tif → $out"
  "$BFCONVERT" -noflat -bigtiff -pyramid-resolutions 4 -pyramid-scale 2 "$tif" "$out"
done
