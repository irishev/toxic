import gensim
import csv
import re
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import json
from torch.autograd import Variable

# Load Google's pre-trained Word2Vec model.
print('Loading word2vec...')
model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/dilab3/Downloads/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True)
print('Done!')

f = open('C:/Users/dilab3/Downloads/train.csv/train.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)

word2ind = {}
wordcount = {}
ind2vec = []

ind2vec.append(np.zeros(300))

for i, line in tqdm(enumerate(rdr)):
    if i!=0:
        words = re.sub("[^\w]", " ",  str.lower(line[1])).split()
        for word in words:
            if word not in word2ind.keys():
                if word in wordcount.keys():
                    if word in model.vocab:
                        word2ind[word] = len(word2ind)+1
                        ind2vec.append(model.wv[word])
                    else:
                        word2ind[word] = len(word2ind)+1
                        ind2vec.append(np.random.randn(300))
                else:
                    wordcount[word] = 0
            wordcount[word] += 1

f = open('C:/Users/dilab3/Downloads/stanfordSentimentTreebank/stanfordSentimentTreebank/datasetSentences.txt', 'r', encoding='utf-8')
rdr = csv.reader(f, delimiter='\t')

for i, line in enumerate(rdr):
    if i!=0:
        words = re.sub("[^\w]", " ",  str.lower(line[1])).split()
        for word in words:
            if word not in word2ind.keys():
                if word in wordcount.keys():
                    if word in model.vocab:
                        word2ind[word] = len(word2ind)+1
                        ind2vec.append(model.wv[word])
                    else:
                        word2ind[word] = len(word2ind)+1
                        ind2vec.append(np.random.randn(300))
                else:
                    wordcount[word] = 0
            wordcount[word] += 1


weight = np.stack(ind2vec)
embedding = nn.Embedding(len(word2ind)+1, 300)
embedding.weight.data.copy_(torch.from_numpy(weight))

torch.save(embedding.state_dict(), 'embed.pkl')

print(embedding(Variable(torch.LongTensor([[0,1,66853]]))))

with open('word2ind.dict', 'w') as file:
     file.write(json.dumps(word2ind))

f.close()
