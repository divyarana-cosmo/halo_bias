import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def data_in_halobin(m0, m1, ljack):
    # getting the data
    dat = pd.read_csv('halo_catalog.csv')
    dat['x'] = dat['x']%2500
    dat['y'] = dat['y']%2500
    dat['z'] = dat['z']%2500
    #assigning jackknife region
    ljacks = ljack
    jlen = 2500/ljacks
    ix = dat['x']//jlen; iy = dat['y']//jlen; iz = dat['z']//jlen

    xjkreg = ix + iy*ljacks + iz*ljacks**2
    njacks = len(np.unique(xjkreg))
    print('njacks=%d'%len(np.unique(xjkreg)))
    idx = (np.log10(dat['Mvir'])>m0) & (np.log10(dat['Mvir'])<m1)
    return dat[idx], xjkreg[idx]



def get_xi(logm0, logm1, ljack):
    dat,xjkreg = data_in_halobin(logm0, logm1, ljack)
    njacks = len(np.unique(xjkreg))

    nhalo = len(xjkreg)/(2500)**3
    rbins = np.logspace(np.log10(4), np.log10(20), 11)
    rad   = 0.5*(rbins[:-1] + rbins[1:])

    RR = 4*np.pi*(0.5*(rbins[:-1] + rbins[1:]))**2 * (rbins[1:] - rbins[:-1]) * nhalo


    from scipy.spatial import cKDTree
    xi = np.zeros(len(rad)*njacks)
    rrad = 0.0 *xi

    # tree over full halo catalog
    htree = cKDTree(np.transpose([dat['x'], dat['y'], dat['z']]), boxsize=2500)

    # intrbins, x0 = jj*len(ra), x1 = (jj+1)* len(ra)

    for jj in range(njacks):
        idx  = (xjkreg!=jj)

        # tree over jackknifed catalog
        jtree = cKDTree(np.transpose([dat['x'][idx], dat['y'][idx], dat['z'][idx]]), boxsize=2500)
        DD = htree.count_neighbors(jtree, rbins, cumulative=True)
        DD = np.diff(DD)/sum(idx)
        x0 = jj*len(DD); x1 = (jj+1)* len(DD)

        xi[x0:x1] = DD/RR - 1
        rrad[x0:x1] = rad
        print(jj, DD/RR - 1)
    #np.loadtxt('./xi_%s_%s.dat'%(logm0,logm1), np.transpose([rrad,xi]))
    return rrad, xi




def get_plots(xx, yy, logM0, logM1):
    rrad = xx; xi = yy

    njacks = sum(rrad==rrad[0])
    print('number of jacks = %d'%njacks)

    rad = np.mean(rrad.reshape((njacks, -1)),  axis=0)
    print(rad)

    from colossus.cosmology import cosmology
    cosmo = cosmology.setCosmology('planck18')
    ximm = cosmo.correlationFunction(rad,0.0)

    bias11 = np.sqrt(xi/np.tile(ximm, njacks))
    bias11 = bias11.reshape((njacks, len(rad)))
    
    yy = np.mean(bias11, axis=0)
    print(yy)
    cov     = np.zeros((len(rad),len(rad)))

    for ii in range(len(rad)):
        for jj in range(len(rad)):
            cov[ii][jj] = np.mean((bias11[:,ii] - yy[ii])*(bias11[:,jj] - yy[jj]))
            cov[ii][jj] = (njacks - 1)*cov[ii][jj]
    
    
    
    np.savetxt('logMh_%s_%s_njks_%d.cov'%(logM0,logM1,njacks),cov)
    yyerr = np.sqrt(np.diag(cov))
    
    plt.subplot(2,2,1)
    plt.errorbar(rad, yy, yerr=yyerr, fmt='.', capsize=3)
    np.savetxt('logMh_%s_%s_njks_%d.dat'%(logM0,logM1,njacks), np.transpose([rad, yy, yyerr]))

    plt.ylim(0.8,1.0)
    plt.xlabel(r'$r \, [{\rm h^{-1}Mpc}]$')
    plt.ylabel(r'$b_{h}$')
    plt.xscale('log')
    plt.title(r'$%s<\log[M_{\rm vir}/({\rm h^{-1}M_{\odot}})] < %s$'%(logM0, logM1))
    
    
    corr = 0.0*cov
    for ii in range(len(rad)):
        for jj in range(len(rad)):
            corr[ii][jj] = cov[ii][jj]*1.0/(yyerr[ii]*yyerr[jj])
    
    
    plt.subplot(2,2,2)
    plt.imshow(corr,cmap='PuOr_r',vmin=-1,vmax=1,origin='lower',aspect='equal')
    plt.xlabel(r'$b_{\rm h}$')
    plt.colorbar()



    plt.subplot(2,2,3)
    for jjx in np.random.randint(1,njacks, size=10):
        x0 = jjx*len(rad); x1 = (jjx+1)* len(rad)
        plt.plot(rad,xi[x0:x1]/xi[0:int(len(rad))])
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(r'$r \, [{\rm h^{-1}Mpc}]$')
    plt.ylabel('ratio')
    #plt.ylabel(r'$\xi(r)$')
    #plt.ylim(0.05,2)


    xi = xi.reshape((njacks, len(rad)))
    
    yy = np.mean(xi, axis=0)
    print(yy)
    cov     = np.zeros((len(rad),len(rad)))

    for ii in range(len(rad)):
        for jj in range(len(rad)):
            cov[ii][jj] = np.mean((xi[:,ii] - yy[ii])*(xi[:,jj] - yy[jj]))
            cov[ii][jj] = (njacks - 1)*cov[ii][jj]
    
    
    
    np.savetxt('logMh_%s_%s_njks_%d_2pf.cov'%(logM0,logM1,njacks),cov)
    yyerr = np.sqrt(np.diag(cov))
    
    plt.subplot(2,2,4)
    
    corr = 0.0*cov
    for ii in range(len(rad)):
        for jj in range(len(rad)):
            corr[ii][jj] = cov[ii][jj]*1.0/(yyerr[ii]*yyerr[jj])
    
    
    plt.imshow(corr,cmap='PuOr_r',vmin=-1,vmax=1,origin='lower',aspect='equal')
    plt.xlabel(r'$\xi(r)$')
    plt.colorbar()





    plt.tight_layout()
    
    plt.savefig('logMh_%s_%s_njks_%d.png'%(logM0,logM1,njacks))


import sys
logm0 = float(sys.argv[1])
logm1 = float(sys.argv[2])
ljack = float(sys.argv[3])

xx,yy = get_xi(logm0, logm1, ljack)
get_plots(xx, yy, logm0, logm1)



