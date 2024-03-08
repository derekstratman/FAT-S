# FAT-S
Feature Alignment Technique for thunderStorms (FAT-S)

FAT-S is a Python program developed by Derek Stratman at OU/CIWRO and NOAA/OAR/NSSL to correct thunderstorm displacement errors in convection-allowing models using composite reflectivity. The code can be modified for different systems and fields. More details about FAT-S can be found in the references below.

References for FAT-S:

Stratman, D. R. and C. K. Potvin, 2022: Testing the Feature Alignment Technique (FAT) in an Ensemble-Based Data Assimilation and Forecast System with Multiple-Storm Scenarios, Mon. Wea. Rev., 150, 2033-2054, https://doi.org/10.1175/MWR-D-21-0289.1.

Stratman, D. R., C. K. Potvin, and L. J. Wicker, 2018: Correcting storm displacement errors in ensembles using the feature alignment technique (FAT). Mon. Wea. Rev., 146, 2125–2145, https://doi.org/10.1175/MWR-D-17-0357.1.

Versions of FAT-S:
1. fats_wrf.py - designed to run with WRF files (wrfout, wrfinput, etc.) and observation files with a matching domain grid
2. More versions to come!

Files to run FAT-S for WRF:
1. fats_wrf.py
2. fats_functions.py
3. state_vector_wrf.py
4. ctables.py

Before running FAT-S for WRF:
1. Adjust listed parameters in top of fats_wrf.py
2. Pick the correct microphysics scheme in fats_wrf.py using the scheme names in state_vector_wrf.py

Running FAT-S for WRF:

python fats_wrf.py -w wrfout_file -o observed_reflectivity.nc -p

where "-p" is a flag to make a 3-panel plot
