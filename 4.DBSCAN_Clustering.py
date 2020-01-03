import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

#IMPORT DATASET
#-------------------------------------------------------------------------------------
FS = []
for i in range(7):
    filename = 'FS'+str(i)+'.csv'
    f = open(filename).readlines()
    f = [j.strip().split(',') for j in f]
    FS.append(np.float64(f))


#DBSCAN CLUSTERING
#-------------------------------------------------------------------------------------
def Normalize(SETS,n):
    LIST = [list(i) for i in SETS]
    clus = [-1]*n
    for i in range(len(LIST)):
        for j in LIST[i]:
            clus[j] = i
    return clus

def DBSCAN(data,DIST,Eps,MinPts,N):
    #MENCARI CORE DAN NONCORE POINT
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

    #MENCARI REACHABLE DAN NOISE DARI NONCORE POINT
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

    #MENCARI NEIGHBORS CORE POINT
    neghcore = []
    for idx_i in core:
        #Cari tetangga core ke-i
        allnegh = np.arange(N)[np.where(DIST[idx_i] <= Eps,True,False)]
        neghcore.append([i for i in allnegh if i in core])

    #MENCARI NEIGHBORS YANG SAMA PADA CORE
    for i,negh_i in enumerate(neghcore):
        for j,negh_j in enumerate(neghcore):
            if i == j:
                continue
            iris = np.intersect1d(negh_i,negh_j)
            if len(iris) > 0:
                neghcore[i] = np.union1d(negh_i,negh_j)
                neghcore[j] = np.union1d(negh_i,negh_j)
            
    setnegh = set(tuple(i) for i in neghcore)
    cluster = Normalize(setnegh,N)

    #print('Core Point     :',core)
    #print('Reachable Point:',reach)
    #print('Noise Point    :',noise)
    return cluster

def PrintNumClus(clus,k):
    for i in range(k):
        print('   - Cluster',i+1,':',list(clus).count(i),'Documents')


#MAIN
#-------------------------------------------------------------------------------------
print('DBSCAN CLUSTERING')

meas = 'cosine'

for x,data in enumerate(FS):
    print('>> UNTUK FS',x)
    DIST = pairwise_distances(data,metric=meas)
    print('   Rerata Jarak Antar Data:',np.average(DIST),'\n')
    
    Eps = input('   Input Eps   : ')
    Eps = np.float(Eps)
    MinPts = input('   Input MinPts: ')
    MinPts = np.int(MinPts)
    print('')
    
    N = len(data)
    cluster = DBSCAN(data,DIST,Eps,MinPts,N)
    PrintNumClus(cluster,max(cluster)+1)
    print('   Jumlah Noise:',cluster.count(-1),'\n')
