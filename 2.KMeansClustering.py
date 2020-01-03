#IMPORT DEPENDENCIES
#-------------------------------------------------------------------------------------
import numpy as np, itertools
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import pairwise_distances as PD
from scipy.stats import rankdata

#IMPORT DATASET
#-------------------------------------------------------------------------------------
FS = []
for i in range(7):
    filename = 'FS'+str(i)+'.csv'
    f = open(filename).readlines()
    f = [j.strip().split(',') for j in f]
    FS.append(np.float64(f))

#CLUSTERING USING K-MEANS
#-------------------------------------------------------------------------------------
def FindInit(data,k):
    D = PD(data)
    npls = k; N = npls+k
    SIMS = 1-(D/np.max(D)) if meas == 'euclidean' else 1-D
    teta = np.average(SIMS)
    negh = np.where(SIMS >= teta,1,0)
    idxc = np.argsort(negh.sum(0),kind='mergesort')
    idxc = idxc[::-1][0:N]
    link,simi = [],[]
    for i,j in itertools.combinations(idxc,2):
        link.append(np.dot(negh[i],negh[j]))
        simi.append(SIMS[i,j])
    rlnk = rankdata(link,method='dense')
    rsim = rankdata(simi,method='dense')
    rsum = squareform(rlnk+rsim)
    rcom,cand,temp = [],[],np.arange(N)
    for i in itertools.combinations(temp,k):
        sect = np.intersect1d(temp,i)
        rank = rsum[sect][:,sect]
        rcom.append(np.sum(rank)/2)
        cand.append([idxc[v] for v in i])
    return sorted(cand[np.argmin(rcom)])

def Diss(A,B):
    return pdist([A,B],metric=meas)[0]

def FindClus(data,cent):
    clus = []
    for i in data:
        temp = []
        for j in cent:
            temp.append(Diss(i,j))
        clus.append(np.argmin(temp))
    return clus

def FindCent(data,clus,k):
    newc = []
    for i in range(k):
        newc.append(np.mean(data[np.isin(clus,i)],0))
    return np.asarray(newc)

def KMeans(data,k):
    initc = FindInit(data,k)
    cent1 = data[initc]
    clus1 = FindClus(data,cent1)

    conv = False
    while not conv:
        cent2 = FindCent(data,clus1,k)
        clus2 = FindClus(data,cent2)
        if clus1 == clus2:
            conv = True
        else:
            clus1 = clus2
    return np.asarray(clus2)

def DaviesBouldin(data,clus):
    k = max(clus)+1
    clus_k = [data[clus==i] for i in range(k)]
    cent_k = [np.mean(i,axis=0) for i in clus_k]
    varian = [np.mean([Diss(p,cent_k[i]) for p in j]) for i,j in enumerate(clus_k)]
    db=[]
    for i in range(k):
        for j in range(k):
            if j != i:
                db.append((varian[i]+varian[j])/Diss(cent_k[i],cent_k[j]))
    return (np.max(db)/k)

def PrintNumClus(clus,k):
    for i in range(k):
        print('   - Cluster',i+1,':',list(clus).count(i),'Documents')

def PrintElementClus(clus,n,k):
    for i in range(k):
        idx = np.arange(n)[clus==i]+1
        print('   - Cluster',i+1,':',ToSTR(idx))
        
def ToSTR(listnum):
    return ', '.join([str(i) for i in listnum])

#MAIN
#-------------------------------------------------------------------------------------
print('K-MEANS CLUSTERING')

N = len(FS[0])
K = 8
meas = 'cosine'

for x,DATA in enumerate(FS):
    print('>> UNTUK FS'+str(x)+':')
    CLUS = KMeans(DATA,K)
    DAVS = DaviesBouldin(DATA,CLUS)
    PrintNumClus(CLUS,K)
    #PrintElementClus(CLUS,N,K)
    print('   Davies Bouldin Index:',DAVS,'\n')
    
