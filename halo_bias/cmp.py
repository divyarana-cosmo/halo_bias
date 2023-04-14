import numpy as np
import matplotlib.pyplot as plt
from glob import glob

flist = np.sort(glob('logMh_12.0_13.0_njks_*.dat'))

plt.subplot(2,2,1)

for fil in flist:
    xx,yy,yyerr = np.loadtxt(fil, unpack=1)
    cov = np.loadtxt(fil.split('dat')[0] + 'cov')
    njacks = int(fil.split('_')[-1].split('.dat')[0])
    print(yyerr)
    print(np.sqrt(np.diag(cov)))
    hartlap = (njacks - len(yy) -1)/(njacks -1)    
    icov = hartlap*np.linalg.inv(cov)
    snr = np.sqrt(np.dot(yy,np.dot(icov,yy)))
    plt.errorbar(xx,yy,yerr=yyerr, fmt='.', capsize=5, label='jk=%s,snr=%2.2f'%(njacks,snr))

plt.xscale('log')

plt.xlabel(r'$r\,[{\rm h^{-1}Mpc}]$')
plt.ylabel(r'$b_{\rm h}$')
plt.legend()
plt.savefig('cmp_plot.png', dpi=300)

