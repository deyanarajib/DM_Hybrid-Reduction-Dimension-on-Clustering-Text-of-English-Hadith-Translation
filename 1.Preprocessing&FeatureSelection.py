#IMPORT DEPENDENCIES
#-------------------------------------------------------------------------------------
from nltk.tokenize import word_tokenize as token
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string, math, numpy as np, pandas as pd

#IMPORT DATASET
#-------------------------------------------------------------------------------------
rawdata = []
for j in range(0,717):
    x = open(str(j+1)+'.txt','r').read()
    rawdata.append(x.replace('\n',' '))

#PRE-PROCESSING
#-------------------------------------------------------------------------------------
print('>> PREPROCESSING...',end=' ')

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

print('Done!')

#TERM WEIGHTING
#-------------------------------------------------------------------------------------
print('>> TERM WEIGHTING...',end=' ')
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

print('Done!')

#FEATURE SELECTION
#-------------------------------------------------------------------------------------
print('>> FEATURE SELECTION...',end=' ')#FS0 >> Tanpa Feature Selection
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

print('Done!')

#SAVING
#-------------------------------------------------------------------------------------
print('>> SAVING TO CSV...',end=' ')
for i in range(7):
    df = pd.DataFrame(eval('FS'+str(i)))
    df.to_csv('FS'+str(i)+'.csv',index=False,header=False)

print('Done!')
