import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

filename = 'FS1.csv'
data = open(filename).readlines()
data = [i.strip().split(',') for i in data]
data = np.float64(data)

meas = 'cosine'
N = len(data)
DIST = pairwise_distances(data,metric=meas)

Epsinit = np.average(DIST)

Epsmins = [Epsinit-((i+1)*0.01) for i in range(1000)]
Epsplus = [Epsinit+((i+1)*0.01) for i in range(1000)]

rangeEps = sorted(Epsmins+[Epsinit]+Epsplus)

for Eps in rangeEps:
    if Eps <= 0:
        continue
    print('>> UNTUK EPS =',Eps)
    for MinPts in range(1,51):
        print('   <> Pada MinPts =',MinPts,end=', ')
    
        core,noncore = [],[]
        for indeks,data_i in enumerate(DIST):
            neghbors_i = np.where(data_i <= Eps,1,0)
            jml_negh_i = neghbors_i.sum()
            if  jml_negh_i >= MinPts:
                #data_i berarti core
                core.append(indeks)
            else:
                #data_i berarti noncore
                noncore.append(indeks)

        reach,noise = [],[]
        for idx_i in noncore:
            close = np.arange(N)[np.where(DIST[idx_i] <= Eps,True,False)]
            check = np.intersect1d(close,core)
            if len(check) != 0:
                #noncore ke-i berarti reachable dari core
                reach.append(idx_i)
            else:
                #noncore ke-i berarti noise
                noise.append(idx_i)

        neghcore = []
        for idx_i in core:
            #Cari tetangga core ke-i
            allnegh = np.arange(N)[np.where(DIST[idx_i] <= Eps,True,False)]
            neghcore.append([i for i in allnegh if i in core])

        for i,negh_i in enumerate(neghcore):
            for j,negh_j in enumerate(neghcore):
                if i == j:
                    continue
                iris = np.intersect1d(negh_i,negh_j)
                if len(iris) > 0:
                    neghcore[i] = np.union1d(negh_i,negh_j)
                    neghcore[j] = np.union1d(negh_i,negh_j)
            
        setnegh = set(tuple(i) for i in neghcore)
        print('Banyak Cluster :',len(setnegh))

        #print('Core Point     :',core)
        #print('Reachable Point:',reach)
        #print('Noise Point    :',noise)
    print('')
