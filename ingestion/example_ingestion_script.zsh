#!/bin/zsh 
for f in /Users/kirkdebaets/projects/customers/nyt/Data/mevadata/bus_331/*avi; do
python -u process_meva_videos.py $f >> mkd.out 2>&1 &
done

