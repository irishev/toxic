import gensim
import csv
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm

# Load Google's pre-trained Word2Vec model.
print('Loading word2vec...')
model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/dilab3/Downloads/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True)
print('Done!')

f = open('C:/Users/dilab3/Downloads/test.csv/test.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
csv.field_size_limit(5000000)
inputs = []
ids = []
for i, line in enumerate(rdr):
    if i!=0:
        words = re.sub("[^\w]", " ",  str.lower(line[1])).split()
        wordlist = []
        for word in words:
            if word in model.vocab:
                wordlist.append(word)
        if len(wordlist) < 32:
            for _ in range(32-len(wordlist)):
                wordlist.append('')
        elif len(wordlist) > 32:
            wordlist = wordlist[0:32]

        inputs.append(wordlist)
        ids.append(line[0])


f.close()

class Conv(nn.Module):
    def __init__(self, in_c, out_c, **kwargs):
        super(Conv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, bias=False, **kwargs),
            nn.BatchNorm2d(out_c),
        )
    def forward(self, x):
        return self.layer1(x)

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            Conv(300, 300, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            Conv(300, 300, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            Conv(300, 300, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(300, 6)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.sigmoid(x)

cnn = CNN()
cnn.load_state_dict(torch.load('cnn_epoch2.pkl'))

cnn.eval()

result = []

f = open('submission.csv', 'w', encoding='utf-8')
wr = csv.writer(f)
wr.writerow(['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'])

# Test the Model
for i in tqdm(range(len(inputs)//64)):
    lists = inputs[i*64:(i+1)*64]
    batch = []
    for words in lists:
        vecs = []
        for word in words:
            if word=='':
                vecs.append(torch.zeros((300, 1, 1)))
            else:
                vecs.append(torch.Tensor(model.wv[word]).view(300, 1, 1))
        batch.append(torch.cat(vecs, dim=1))
    input = torch.stack(batch, dim=0)

    input = Variable(input)
    outputs = cnn(input)
    for j in range(64):
        wr.writerow([ids[i*64+j],outputs.data[j][0],outputs.data[j][1],outputs.data[j][2],outputs.data[j][3],outputs.data[j][4],outputs.data[j][5]])

lists = inputs[(len(inputs)//64)*64:len(inputs)]
batch = []
for words in lists:
    vecs = []
    for word in words:
        if word=='':
            vecs.append(torch.zeros(300, 1, 1))
        else:
            vecs.append(torch.Tensor(model.wv[word]).view(300, 1, 1))
    batch.append(torch.cat(vecs, dim=1))
input = torch.stack(batch, dim=0)

input = Variable(input)
outputs = cnn(input)
for j in range(outputs.size(0)):
    wr.writerow([ids[(len(inputs)//64)*64+j],outputs.data[j][0],outputs.data[j][1],outputs.data[j][2],outputs.data[j][3],outputs.data[j][4],outputs.data[j][5]])


f.close()
