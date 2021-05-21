# instantaneousISC-col786
iisc_final.py runnable on python-3

getLowPVoxels.py to get the lowest P-vals map summed across the time series

getLowPVoxelstimewise.py to get the lowest P-vals map at each time point (4-d map)

An example run would look like
python iisc_final.py -i \*.nii.gz -o results -t 2 -n -b 0 1 2
