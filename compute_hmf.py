#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from colossus.cosmology import cosmology
cosmology.setCosmology('planck15');
from colossus.lss import mass_function

# hmf measuring
def get_hmf(mharr, lbox):
    bins = np.linspace(7,15,101) # log Mvir bins
    hist, binedgs = np.histogram(np.log10(mharr), bins=bins)
    xx = (binedgs[1:] + binedgs[:-1])*0.5
    return xx, hist/(lbox)**3/ (bins[1]-bins[0])

from glob import glob
flist = glob('DataStore/TNG300-1-Dark/fof_subhalo_tab_099.*')

hmf =   0.0
xx  =   0.0
import h5py
for fil in flist:
    print(fil)
    try:
        f = h5py.File(fil,'r')
        #xx, hh = get_hmf(f['Group']['Group_M_Mean200'][:]*1e10, lbox=300)
        xx, hh = get_hmf(f['Subhalo']['SubhaloMass'][:]*1e10, lbox=300)
        hmf += hh
    except:
        print('sucks')


plt.subplot(3,3,1)
plt.plot(xx, hmf, '.')
mfunc = mass_function.massFunction(10**xx, 0.0, mdef = 'vir', model = 'tinker08', q_out = 'dndlnM')
plt.plot(xx, mfunc/np.log10(np.e))
plt.yscale('log')
plt.xlabel(r'$\log M_{\rm h}$')
plt.ylabel(r'$n(M_{\rm h})$')
plt.savefig('hmf.png', dpi=300)
