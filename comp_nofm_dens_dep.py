import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/rana/github_0/aum/install0_dens_dep/lib/python3.13/site-packages')
import cosmology as cc
a = cc.cosmology(0.27,0.0,-1.0,0.0,0.0476,0.7,2.726,0.8,0.96,np.log10(8.0),1.0)

xx = 10**np.linspace(9,16,20)
yy = np.array([])

for mm in xx:
    yy = np.append(yy,a.nofm(mm,0.0))

plt.subplot(2,2,1)
plt.plot(xx, yy, '.', label='Tinker10')

delta_thres = np.linspace(-1,1,5)
for ff in delta_thres:
    yy = np.array([])
    for mm in xx:
        yy = np.append(yy,a.MF_TI10_dens_dep(mm,0.0,ff))
    plt.plot(xx, yy, label=r'$\delta_{\rm thres} = %2.2f$'%ff)

plt.xlabel(r'$M_{\rm h}$')
plt.ylabel(r'$n(M_{\rm h})$')

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('test.png', dpi=300)



