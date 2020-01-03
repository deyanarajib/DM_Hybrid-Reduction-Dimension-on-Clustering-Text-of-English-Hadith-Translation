raws = []
for i in range(100):
    f = open(str(i+1)+'.txt').read()
    raws.append(f.strip())

#PREPROCESSING
import numpy as np
from string import punctuation as punct
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize as wt

punct   = list(punct)+["''",'--','...','``']
swords  = stopwords.words('english') + punct
stemmer = SnowballStemmer('english')

docs = []
for doc in raws:
    temp = []
    for word in wt(doc):
        word = word.replace("'",'')
        if word in swords or len(word) <= 2:
            continue
        temp.append(stemmer.stem(word))
    docs.append(temp)

BOG = sorted(set(np.concatenate(docs)))

#FEATURE EXTRACTION
