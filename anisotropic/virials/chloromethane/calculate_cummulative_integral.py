#!/usr/bin/env python
"""

Last Updated:
"""
import numpy as np
import scipy as sp
import sys
from simtk import unit
from scipy.signal import savgol_filter

# Read in Virials
ofile = 'fmayer.out'
with open(ofile,'r') as f:
    data = [line.split() for line in f.readlines()]
    data = np.array(data,dtype=np.float)
    r_tot = data[:,0]
    fmayer_tot = data[:,1:]

# Correct virials for -infinity energies
thresh = 1e5
fmayer_tot = np.where( fmayer_tot > thresh, -1, fmayer_tot)

# Apply a Savitzky-Golay filter to remove noise in the data
print fmayer_tot.shape
#fmayer_tot = sp.signal.savgol_filter(fmayer_tot,window_length=10,polyorder=3,axis=1)
fmayer_tot = savgol_filter(fmayer_tot,window_length=25,polyorder=3,axis=0)

print type(fmayer_tot)

print r_tot[0]
print fmayer_tot[0]
dr_virial = r_tot[1] - r_tot[0]
            
# Numerically integrate along r to obtain virial
for i in xrange(len(r_tot)):
    virial = np.trapz(fmayer_tot[:i]*r_tot[:i,np.newaxis]**2, x=r_tot[:i], dx=dr_virial, axis=0)
    nm3_to_cm3 = 1e-21
    avogadro = 6.02214086e23
    virial *= -2*np.pi*avogadro*nm3_to_cm3
    print i, r_tot[i], virial[7]

## print type(virial)
## # Convert into units of cm^3/mol (factor of -2pi comes from 4pi in
## # integrand and -1/2 factor outside of integral)
## nm3_to_cm3 = 1e-21
## avogadro = 6.02214086e23
## virial *= -2*np.pi*avogadro*nm3_to_cm3
## 
## print 'Virials are ' , virial
## 
## with open('new_fmayer.out','w') as f:
##     template = '{:16.8f}'*((fmayer_tot.shape)[1] + 1) + '\n'
##     for r,line in zip(r_tot,fmayer_tot):
##         # f.write(template.format(line*unit.mole))
##         f.write(template.format(r,*line))
## 
## with open('new_virials.dat','w') as f:
##     template = '{:16.8f}\n'
##     for line in virial:
##         # f.write(template.format(line*unit.mole))
##         f.write(template.format(line))
