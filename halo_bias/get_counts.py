

def xi_Mhbin(logM0, logM1):
    # getting the data
    dat = pd.read_csv('halo_catalog.csv')
    dat['x'] = dat['x']%2500
    dat['y'] = dat['y']%2500
    dat['z'] = dat['z']%2500
    #assigning jackknife region
    ljacks = 4
    jlen = 2500/ljacks
    ix = dat['x']//jlen; iy = dat['y']//jlen; iz = dat['z']//jlen
    
    xjkreg = ix + iy*ljacks + iz*ljacks**2
    
    njacks = len(np.unique(xjkreg))
    print('njacks=%d'%njacks)
    
    nhalo = len(ix)/(2500)**3
    rbins = np.logspace(np.log10(4), np.log10(20), 11)
    rad   = 0.5*(rbins[:-1] + rbins[1:])
    
    RR = 4*np.pi*(0.5*(rbins[:-1] + rbins[1:]))**2 * (rbins[1:] - rbins[:-1]) * nhalo
    
    
    from scipy.spatial import cKDTree
    xi = np.zeros(len(rad)*njacks)
    
    # intrbins, x0 = jj*len(ra), x1 = (jj+1)* len(ra)
    
    for jj in range(njacks):
        idx  = (xjkreg!=jj)
        
        htree = cKDTree(np.transpose([dat['x'][idx], dat['y'][idx], dat['z'][idx]]), boxsize=2500)
        DD = htree.count_neighbors(htree, rbins, cumulative=True)
        DD = np.diff(DD)/sum(idx)
        x0 = jj*len(DD); x1 = (jj+1)* len(DD)
        
        xi[x0:x1] = DD/RR - 1
        print(jj)
    return xi
