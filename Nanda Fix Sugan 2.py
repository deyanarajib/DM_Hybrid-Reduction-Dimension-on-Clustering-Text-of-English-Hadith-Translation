#IMPORT DEPENDENCIES
#-------------------------------------------------------------------------------------
from nltk.tokenize import word_tokenize as token
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string, math, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA as PCA_
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

def line():
    [print('-',end='') for i in range(50)]; print('')

#IMPORT DATASET
#-------------------------------------------------------------------------------------
print('1. IMPORT DATASET'); line()
DATA_RAW = []
for j in range(0,717):
    f = open(str(j+1)+'.txt','r').read()
    DATA_RAW.append(f.replace('\n',' '))

concat = np.concatenate([token(i) for i in DATA_RAW])
N = len(DATA_RAW)
V_RAW = len(set(concat))

print('NUMBER OF DOCUMENTS:',N)
print('NUMBER OF FEATURE  :',V_RAW,'\n')

#PRE-PROCESSING
#-------------------------------------------------------------------------------------
print('2. PREPROCESSING'); line()

swords = stopwords.words('english')+list(string.punctuation)
stemmer = SnowballStemmer('english')
junk = tuple(string.punctuation)+tuple([str(k) for k in range(10)])+tuple('¿')

DATA_PREPRO = []
for i in DATA_RAW:
    temp = []
    for j in token(i):
        t = stemmer.stem(str.lower(j))
        if t not in swords and len(t) > 2 and not t.startswith(junk):
            temp.append(t)
    DATA_PREPRO.append(temp)

concat = np.concatenate(DATA_PREPRO)
V_PREPRO = len(set(concat))

print('NUMBER OF DOCUMENTS:',N)
print('NUMBER OF FEATURE  :',V_PREPRO,'\n')

#TERM WEIGHTING
#-------------------------------------------------------------------------------------
print('3. TERM WEIGHTING'); line()
dictionary = list(sorted(set(np.concatenate(DATA_PREPRO))))
            
TF = []
for i in DATA_PREPRO:
    TF.append([i.count(j)/len(i) for j in dictionary])

DF = []
for i in dictionary:
    count=0
    for j in DATA_PREPRO:
        if i in j:
            count += 1
    DF.append(count)

IDF = np.log(N/np.asarray(DF))

TFIDF = np.array([i*IDF for i in TF])

print('NUMBER OF DOCUMENTS:',N)
print('NUMBER OF FEATURE  :',V_PREPRO,'\n')

#FEATURE SELECTION
#-------------------------------------------------------------------------------------
print('4. FEATURE SELECTION USING PCA'); line()

half = int(V_PREPRO/2)

#Document Frequency
mindf = np.argsort(DF)[0:half]
maxdf = [i for i in range(V_PREPRO) if i not in mindf]

#Term Variance
varused = np.copy(TF) #ganti ku TFIDF mun deuk make TFIDF

avrg =  np.average(varused,0) 
sumavrg = []
for i in varused:
    sumavrg.append([math.pow(k,2) for k in i-avrg])
tv = np.average(sumavrg,0)
mintv = np.argsort(tv)[0:half]
maxtv = [i for i in range(V_PREPRO) if i not in mintv]

#Modified Union
c1 = 20/100
c2 = 80/100
u1 = np.random.choice(maxdf,int(c1*half),replace=False)
u2 = np.random.choice(maxtv,int(c1*half),replace=False)
union = np.union1d(u1,u2)

i1 = np.random.choice(maxdf,int(c2*half),replace=False)
i2 = np.random.choice(maxtv,int(c2*half),replace=False)
inter = np.intersect1d(i1,i2)

mix = np.union1d(union,inter)
modified = TFIDF[:,mix]

#PCA
covariance = np.cov(modified.T)
[eigen,v_eigen] = np.linalg.eigh(covariance)
v_row = v_eigen.T[list(reversed(np.argsort(eigen)))]
idxpca = np.matmul(v_row,modified.T)
selfeat = int((30*len(idxpca))/100)
PCA = idxpca[0:selfeat].T

V_PCA = len(PCA[0])

print('   NUMBER OF ORIGINAL FEATURE:',V_PREPRO)
print('   NUMBER OF PCA FEATURE     :',V_PCA,'\n')

#CLUSTERING
#-------------------------------------------------------------------------------------
import itertools
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics.pairwise import pairwise_distances as PD
from scipy.stats import rankdata

def FindInit(data,D,k):
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

def diss(A,B):
    return pdist([A,B],metric=meas)[0]

def FindClus(data,cent):
    clus=[]
    for i in data:
        clus.append(np.argmin([diss(i,j) for j in cent]))
    return np.asarray(clus)

def FindCent(data,clus):
    k = max(clus)+1
    cent=[]
    for i in range(k):
        cent.append(np.mean(data[clus==i],0))
    return np.asarray(cent)

def KMeans(data,initc):
    cent1 = data[initc]
    clus1 = FindClus(data,cent1)
    conv = False
    while not conv:
        cent2 = FindCent(data,clus1)
        clus2 = FindClus(data,cent2)
        if list(clus1) == list(clus2):
            conv = True
        else:
            clus1 = clus2
    return clus2

def DB_SCAN(data,Eps,Pts):
    DB = DBSCAN(eps = Eps,
                min_samples = Pts,
                metric = meas).fit(data)
        
    clus = DB.labels_
    return clus

#DAVIES BOULDIN
#-------------------------------------------------------------------------------------
def DaviesBouldin(data,clus,k):
    clus_k = [data[clus==i] for i in range(k)]
    cent_k = [np.mean(i,axis=0) for i in clus_k]
    varian = [np.mean([diss(p,cent_k[i]) for p in j]) for i,j in enumerate(clus_k)]
    db=[]
    for i in range(k):
        for j in range(k):
            if j != i:
                db.append((varian[i]+varian[j])/diss(cent_k[i],cent_k[j]))
    return (np.max(db)/k)

#PRINT
#-------------------------------------------------------------------------------------

def PrintNumClus(clus,k):
    for i in range(k):
        print('   - Cluster',i+1,':',list(clus).count(i),'Documents')
        
def PrintElementClus(clus,n,k):
    for i in range(k):
        idx = np.arange(n)[clus==i]+1
        print('   - Cluster',i+1,':',ToSTR(idx))
        
def ToSTR(listnum):
    return ', '.join([str(i) for i in listnum])

def k_distances(x,k):
    dim0 = x.shape[0]
    dim1 = x.shape[1]
    p = PD(x,metric=meas)
    p.sort(axis=1)
    p = p[:,:k]
    pm = p.flatten()
    pm = np.sort(pm)
    return p,pm

def Plot2d(data,k,clus,title,noise):
    for i in range(k):
        x = data[np.isin(clus,i)][:,0]
        y = data[np.isin(clus,i)][:,1]
        plt.scatter(x,y,label='Cluster'+str(i+1))
    if noise:
        x = data[np.isin(clus,-1)][:,0]
        y = data[np.isin(clus,-1)][:,1]
        plt.scatter(x,y,label='Noise')
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()

def Plot3d(data,k,clus,title,noise):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    for i in range(k):
        x = data[np.isin(clus,i)][:,0]
        y = data[np.isin(clus,i)][:,1]
        z = data[np.isin(clus,i)][:,2]
        ax.scatter(x,y,z,label='Cluster'+str(i+1))
    if noise:
        x = data[np.isin(clus,-1)][:,0]
        y = data[np.isin(clus,-1)][:,1]
        z = data[np.isin(clus,-1)][:,2]
        ax.scatter(x,y,z,label='Noise')
    plt.legend()
    plt.title(title)
    plt.show()


#MAIN
#-------------------------------------------------------------------------------------
print('5. CLUSTERING'); line()

meas    = 'cosine'
pca1 = PCA_(n_components=2)
pca2 = PCA_(n_components=3)
methods = ['KMeans','DBSCAN']
fsname  = ['TANPA REDUKSI','PCA']

print('PROPERTIES!!!')
print('- Number of Documents:',N)
print('- Distances Type     :',meas,'\n')

for i,data in enumerate([TFIDF,PCA]): 
    v = len(data[0])
    print('UNTUK DATA '+fsname[i]+':',v,'Features')

    DIST = PD(data,metric=meas)

    print('<> Using KMeans')
    K_KM = 8 #Jumlah Cluster KMeans
    clusKM = KMeans(data,FindInit(data,DIST,K_KM))

    PrintNumClus(clusKM,K_KM)
    #PrintElementClus(clusKM,N,K_KM)
    
    DBI = DaviesBouldin(data,clusKM,K_KM)
    SIL = silhouette_score(data,clusKM,metric=meas)
    print('   Davies Bouldin Index:',DBI,)
    print('   Silhouette Score    :',SIL,'\n')

    Noise = False
    Plot2d(pca1.fit_transform(data),K_KM,clusKM,fsname[i]+' KMEANS',Noise)
    Plot3d(pca2.fit_transform(data),K_KM,clusKM,fsname[i]+' KMEANS',Noise)

    print('<> Using DBSCAN')
    K_DB = 0 #Inisial Jumlah Cluster DBSCAN
    while K_DB <= 1:
        Pts = 8
        m,m2 = k_distances(data,Pts)
        plt.plot(m2)
        plt.ylabel('k-distances')
        plt.grid(True)
        plt.show()
        Eps = input('   Input Eps: '); Eps = float(Eps)
        clusDB = DB_SCAN(data,Eps,Pts)
        K_DB = max(clusDB)+1
                
        print('   Jumlah Noise:',list(clusDB).count(-1))
        
        if K_DB <= 1:
            print('')
            print('   Karena Jumlah Cluster <= 1, Davies Tidak Bisa Dihitung')
            print('   Input Ulang Eps dan MinPts\n')
        else:
            PrintNumClus(clusDB,K_DB)
            #PrintElementClus(clusDB,N,K_DB)
            
            DBI = DaviesBouldin(data,clusDB,K_DB)
            SIL = silhouette_score(data,clusKM,metric=meas)
            print('   Davies Bouldin Index:',DBI,)
            print('   Silhouette Score    :',SIL,'\n')

            Noise = False #True keun mun deuk nampilkeun noise
            Title = fsname[i]+' DBSCAN' if not Noise else fsname[i]+' DBSCAN+NOISE'
            Plot2d(pca1.fit_transform(data),K_DB,clusDB,Title,Noise)
            Plot3d(pca2.fit_transform(data),K_DB,clusDB,Title,Noise)
