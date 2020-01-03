#IMPORT DEPENDENCIES
#-------------------------------------------------------------------------------------
from nltk.tokenize import word_tokenize as token
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string, math, numpy as np, pandas as pd

def line():
    [print('-',end='') for i in range(50)]; print('')

#IMPORT DATASET
#-------------------------------------------------------------------------------------
print('1. IMPORT DATASET'); line()
rawdata = []
for j in range(0,717):
    x = open(str(j+1)+'.txt','r').read()
    rawdata.append(x.replace('\n',' '))

dtoken = [token(i) for i in rawdata]
concat = np.concatenate(dtoken)

print('NUMBER OF DOCUMENTS:',len(rawdata))
print('NUMBER OF FEATURE  :',len(set(concat)),'\n')

#PRE-PROCESSING
#-------------------------------------------------------------------------------------
print('2. PREPROCESSING'); line()

stop_word = stopwords.words('english')+list(string.punctuation)
stemmer = SnowballStemmer('english')
junk = tuple(string.punctuation)+tuple([str(k) for k in range(10)])+tuple('¿')

doc=[]
for i in rawdata:
    temp=[]
    for j in token(i):
        word = stemmer.stem(str.lower(j))
        if word not in stop_word and len(word) > 2 and not word.startswith(junk):
            temp.append(word)
    doc.append(temp)

concat = np.concatenate(doc)
print('NUMBER OF DOCUMENTS:',len(rawdata))
print('NUMBER OF FEATURE  :',len(set(concat)),'\n')

#TERM WEIGHTING
#-------------------------------------------------------------------------------------
print('3. TERM WEIGHTING'); line()
dictionary=[]
for i in doc:
    for j in i:
        if j not in dictionary:
            dictionary.append(j)
            
tf=[]
for i in doc:
    temp=[]
    for j in dictionary:
        temp.append(i.count(j))
    tf.append(temp)

df=[]
for i in dictionary:
    count=0
    for j in doc:
        if i in j:
            count+=1
    df.append(count)

idf=[]
for i in df:
    idf.append(math.log10(len(doc)/i))

print('NUMBER OF DOCUMENTS:',len(rawdata))
print('NUMBER OF FEATURE  :',len(idf),'\n')

#FEATURE SELECTION
#-------------------------------------------------------------------------------------
print('4. FEATURE SELECTION'); line()
    
#FS0 >> Tanpa Feature Selection
FS0 = np.array([np.array(i)*np.array(idf) for i in tf])

#FS1 >> Using DF
halfterms = int((len(df)-len(df)%2)/2)
idxmin1 = np.argsort(df)[0:halfterms]
idxmax1 = list(set(np.arange(len(df)))-set(idxmin1))
FS1 = np.delete(FS0,idxmin1,1)

#FS2 >> Using TV
avg = np.average(FS0,0)
sumtfidf = []
for i in FS0:
    sumtfidf.append([math.pow(k,2) for k in i-avg])
tv = np.average(sumtfidf,0)
idxmin2 = np.argsort(tv)[0:halfterms]
idxmax2 = list(set(np.arange(len(df)))-set(idxmin2))
FS2 = np.delete(FS0,idxmin2,1)

#FS3 >> Using DF union TV
idxunion = np.union1d(idxmax1,idxmax2)
notunion = list(set(np.arange(len(df)))-set(idxunion))
FS3 = np.delete(FS0,notunion,1)

#FS4 >> Using DF intersect TV
idxintersect = np.intersect1d(idxmax1,idxmax2)
notintersect = list(set(np.arange(len(df)))-set(idxintersect))
FS4 = np.delete(FS0,notintersect,1)

#FS5 >> Using Modified Union
C1 = 20/100
C2 = 80/100
U1 = np.random.choice(idxmax1,round(C1*len(idxmax1)),replace=False)
U2 = np.random.choice(idxmax2,round(C1*len(idxmax2)),replace=False)
Uni = np.union1d(U1,U2)
I1 = np.random.choice(idxmax1,round(C2*len(idxmax1)),replace=False)
I2 = np.random.choice(idxmax2,round(C2*len(idxmax2)),replace=False)
Int = np.intersect1d(I1,I2)
idxmix = np.union1d(Uni,Int)
FS5 = np.delete(FS0,list(set(np.arange(len(df)))-set(idxmix)),1)

#FS6 >> PCA
DT = FS5.T;
C = np.cov(DT)
[Eigen,VEigen] = np.linalg.eigh(C)
VT = VEigen.T
VRow = VT[list(reversed(np.argsort(Eigen)))]
PCA = np.matmul(VRow,DT)
selectedfeat = int(30*len(PCA)/100)
FS6 = PCA[0:selectedfeat].T

namefs =  ['Original Feature',
           'Document Frequency',
           'Term Variance',
           'DF union TV',
           'DF intersect TV',
           'Modified Union',
           'PCA']

for i,FS in enumerate([FS1,FS2,FS3,FS4,FS5,FS6]):
    print('>> FS'+str(i+1)+':',namefs[i+1])
    print('   Number of Feature:',len(FS[0]),'\n')

#K-MEANS CLUSTERING
#-------------------------------------------------------------------------------------
import itertools
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics.pairwise import pairwise_distances
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

def KMeans(data,D):
    k = 8
    initc = FindInit(data,D,k)
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

#DBSCAN CLUSTERING
#-------------------------------------------------------------------------------------
def Normalize(SETS,n):
    LIST = [list(i) for i in SETS]
    clus = [-1]*n
    for i in range(len(LIST)):
        for j in LIST[i]:
            clus[j] = i
    return np.array(clus)

def DBSCAN(data,DIST):
    N = len(data)
    print('   Rerata Jarak Antar Data:',np.average(DIST))
    Eps    = input('   Input Eps              : ')
    MinPts = input('   Input MinPts           : ')
    Eps    = np.float(Eps)
    MinPts = np.int(MinPts)
    
    core,noncore = [],[]
    for indeks,data_i in enumerate(DIST):
        neghbors_i = np.where(data_i <= Eps,1,0)
        jml_negh_i = neghbors_i.sum()
        if  jml_negh_i >= MinPts:
            core.append(indeks)
        else:
            noncore.append(indeks)

    reach,noise = [],[]
    for idx_i in noncore:
        close = np.arange(N)[np.where(DIST[idx_i] <= Eps,True,False)]
        check = np.intersect1d(close,core)
        if len(check) != 0:
            reach.append(idx_i)
        else:
            noise.append(idx_i)

    neghcore = []
    for idx_i in core:
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
    cluster = Normalize(setnegh,N)
    return cluster

#DAVIES BOULDIN
#-------------------------------------------------------------------------------------
def DaviesBouldin(data,clus,k):
    clus_k = [data[clus==i] for i in range(k)]
    cent_k = [np.mean(i,axis=0) for i in clus_k]
    varian = [np.mean([Diss(p,cent_k[i]) for p in j]) for i,j in enumerate(clus_k)]
    db=[]
    for i in range(k):
        for j in range(k):
            if j != i:
                db.append((varian[i]+varian[j])/Diss(cent_k[i],cent_k[j]))
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


#MAIN
#-------------------------------------------------------------------------------------
print('5. CLUSTERING'); line()

N = len(FS0)
meas = 'cosine'
methods = ['KMeans','DBSCAN']

print('PROPERTIES!!!')
print('- Number of Documents:',N)
print('- Clustering Methods :',' and '.join(methods))
print('- Distances Type     :',meas,'\n')

for i,data in enumerate([FS0,FS1,FS2,FS3,FS4,FS5,FS6]):
    v = len(data[0])
    print('For FS'+str(i)+':','('+namefs[i]+')',v,'Features')

    DIST = pairwise_distances(data,metric=meas)
    
    for meth in methods:
        print('>> Using',meth)

        ulang = True
        while ulang:
            ulang = False
            
            clus = eval(meth)(data,DIST)
            K = max(clus)+1

            if meth == 'DBSCAN':
                print('   Jumlah Noise           :',list(clus).count(-1))
        
            PrintNumClus(clus,K)
            #PrintElementClus(clus,N,K)

            if K <= 1:
                print('')
                print('   Karena Jumlah Cluster <= 1, Davies Tidak Bisa Dihitung')
                print('   Input Ulang Eps dan MinPts\n')
                ulang = True
            else:
                DBI = DaviesBouldin(data,clus,K)
                print('   Davis Bouldin Index:',DBI,'\n')
