import pandas as pd, numpy as np
from nltk.tokenize import word_tokenize as wt
from string import punctuation as punct
from scipy.linalg import svd
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances as pair
from nltk.corpus import stopwords

punct = list(punct)+["''",'--','...','``']
swords = stopwords.words('english')+["'s"]

titles = []
for i in range(100):
    f = open(str(i+1)+'.txt').read()
    titles.append(f.strip())
    
print('>> PART1: CREATING COUNT MATRIX:')
docs = []
for doc in titles:
    doc = doc.lower().replace("'",'')
    temp = []
    for word in wt(doc):
        if word in swords or word in punct:
            continue
        temp.append(word)
    docs.append(temp)

BOG = sorted(set(np.concatenate(docs)))

N = len(docs)
V = len(BOG)

TF1   = np.asarray([[i.count(j) for j in BOG] for i in docs])
TF2   = np.asarray([[i.count(j)/len(i) for j in BOG] for i in docs])
DF    = np.asarray([sum(np.where(np.asarray(i)!=0,1,0)) for i in TF2.T])
IDF   = np.asarray([np.log10(N/i) for i in DF])
TFIDF = np.asarray([IDF*i for i in TF2])

indx  = np.arange(V)[DF>2]
term  = np.asarray(BOG)[indx]
#A = pd.DataFrame(TF1.T[indx])
#A.columns = ['T'+str(i+1) for i in range(N)]
#A.index   = ['   '+i for i in term]
#print(A,'\n')

print('>> PART2: MODIFY COUNT MATRIX WITH TFIDF:')

#W = pd.DataFrame(np.round(TFIDF.T[indx],2))
#W.columns = ['W'+str(i+1) for i in range(N)]
#W.index   = ['   '+i for i in term]
#print(W,'\n')


print('>> PART3: SINGULAR VALUE DECOMPOSITION\n')

n = 100

a,b,c = [-1*np.round(i,2) for i in svd(TF1.T[indx])]
A = a[:,0:n]
#Adf = pd.DataFrame(A)
#Adf.columns = ['']*n
#Adf.index   = ['   '+i for i in term]
#print('   Matrix A:')
#print(Adf,'\n')

B = np.zeros((n,n))
np.fill_diagonal(B,b[0:n])
#Bdf = pd.DataFrame(B)
#Bdf.columns = ['']*n
#Bdf.index   = ['   ']*n
#print('   Matrix B:')
#print(Bdf,'\n')

C = c[0:n]
Cdf = pd.DataFrame(C)
#Cdf.columns = ['']*N
#Cdf.index   = ['   ']*n
#print('   Matrix C:')
#print(Cdf,'\n')

print('>> PART4: CLUSTERING BY COLOR\n')

plt.imshow(C,cmap='seismic',vmin=-1,vmax=1)
plt.colorbar()
#plt.xticks(np.arange(N), ['T'+str(i+1) for i in range(N)])
#plt.yticks(np.arange(n), ['Dim'+str(i+1) for i in range(n)])
plt.show()

color = np.where(C>=0,'R','B')

c = ['R','B']
dims = []
for x,i in enumerate(color):
    temp = []
    for j in c:
        res = np.arange(N)[np.isin(i,j)]
        temp.append(res)
    dims.append(temp)

Z = []
for i in c:
    for j in c:
        for k in c:
            temp=[]
            for idx,v in enumerate(color.T):
                if all(np.asarray([i,j,k]) == v):
                    temp.append(idx+1)
            if temp == []:
                continue
            Z.append([i,j,k,','.join([str(x) for x in temp])])

Z = pd.DataFrame(Z)
Z.columns = ['Dim'+str(i+1) for i in range(n)]+['Title']
Z.index   = ['   ']*len(Z)
print(Z,'\n')

print('>> PART5: CLUSTERING BY VALUE\n')

data = A[:,1::]
DB = DBSCAN(eps = 0.2,
            min_samples = 2,
            metric = 'euclidean').fit(data)
clus = DB.labels_

for i in range(3):
    print('   Cluster',i+1,':',', '.join(term[np.isin(clus,i)]))
print('   Noise     :',', '.join(term[np.isin(clus,-1)]))
    
for name,(x,y) in enumerate(A[:,1::]):
    plt.scatter(x,y,c='r')
    plt.annotate(term[name],(x,y))
for name,(x,y) in enumerate(C.T[:,1::]):
    plt.scatter(x,y,c='b')
    plt.annotate('T'+str(name+1),(x,y))
plt.xlabel('Dimnesion 2')
plt.ylabel('Dimnesion 3 ')
plt.show()
