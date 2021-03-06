import csv
import re
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import json

json1_file = open('word2ind.dict')
json1_str = json1_file.read()
word2ind = json.loads(json1_str)


f = open('C:/Users/dilab3/Downloads/train.csv/train.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)

dataset = []
for i, line in enumerate(rdr):
    if i!=0:
        words = re.sub("[^\w]", " ",  str.lower(line[1])).split()
        wordlist = []
        for word in words:
            if word in word2ind.keys():
                wordlist.append(word2ind[word])
            else:
                wordlist.append(0)
        if len(wordlist) < 64:
            for _ in range(64-len(wordlist)):
                wordlist.append(0)
        elif len(wordlist) > 64:
            wordlist = wordlist[0:64]

        if sum(wordlist)==0:
            print(line[1])
            continue

        target = line[2:]
        target = list(map(float, target))

        dataset.append((wordlist,target))

f.close()

f = open('C:/Users/dilab3/Downloads/stanfordSentimentTreebank/stanfordSentimentTreebank/datasetSentences.txt', 'r', encoding='utf-8')
rdr = csv.reader(f, delimiter='\t')
f = open('C:/Users/dilab3/Downloads/stanfordSentimentTreebank/stanfordSentimentTreebank/sentiment_labels.txt', 'r', encoding='utf-8')
rdr2 = csv.reader(f, delimiter='|')

sst = []
for i, (line, line2) in enumerate(zip(rdr, rdr2)):
    if i!=0:
        words = re.sub("[^\w]", " ",  str.lower(line[1])).split()
        wordlist = []
        for word in words:
            if word in word2ind.keys():
                wordlist.append(word2ind[word])
            else:
                wordlist.append(0)
        if len(wordlist) < 64:
            for _ in range(64-len(wordlist)):
                wordlist.append(0)
        elif len(wordlist) > 64:
            wordlist = wordlist[0:64]

        if sum(wordlist)==0:
            print(line[1])
            continue

        target = [line2[1]]
        target = list(map(float, target))

        sst.append((wordlist, target))


class Conv(nn.Module):
    def __init__(self, in_c, out_c, **kwargs):
        super(Conv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, bias=False, **kwargs),
            #nn.BatchNorm2d(out_c),
        )
    def forward(self, x):
        return self.layer1(x)

class DCNModule(nn.Module):
    def __init__(self, in_c, mid_c, c):
        super(DCNModule,self).__init__()
        self.layer1 = nn.Sequential(
            Conv(in_c, mid_c, kernel_size=1),
            nn.ReLU(),
            Conv(mid_c, mid_c, kernel_size=(3,1), padding=(1,0), groups=c),
            nn.ReLU(),
            Conv(mid_c, in_c, kernel_size=1)
        )

    def forward(self, x):
        return F.relu(self.layer1(x)+x)

class Downsample(nn.Module):
    def __init__(self, in_c, out_c, c):
        super(Downsample,self).__init__()
        self.layer1 = nn.Sequential(
            Conv(in_c, out_c, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
        )

    def forward(self, x):
        return self.layer1(x)

embedding = nn.Embedding(len(word2ind)+1, 300)
embedding.load_state_dict(torch.load('embed.pkl'))

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.embed = embedding
        self.conv = nn.Sequential(
            # nn.Dropout(),
            Conv(1, 128, kernel_size=(3, 300), stride=(2, 1)),
            nn.ReLU(),
            DCNModule(128, 64, 16),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            DCNModule(128, 64, 16),
            nn.AdaptiveMaxPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128, 6)
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128, 1)
        )

    def forward(self, x, mode):
        x = self.embed(x).unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        if mode==1:
            x = self.fc(x)
        if mode==2:
            x = self.fc2(x)
        return x

cnn = CNN()

learning_rate = 0.001
num_epochs = 20

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=1e-5)
open("loss.txt", 'w')
mean_loss = 0.2

random.shuffle(dataset)
trainset = dataset[:len(dataset)*9//10]
testset = dataset[len(dataset)*9//10:]
# Train the Model
for epoch in range(num_epochs):
    pbar = tqdm(range(len(trainset)//64))
    random.shuffle(trainset)
    for i in pbar:
        lists = trainset[i*64:(i+1)*64]
        batch = []
        targets = []
        for words, target in lists:
            batch.append(torch.LongTensor(words))
            targets.append(torch.FloatTensor(target))

        input = Variable(torch.stack(batch, dim=0))
        label = Variable(torch.stack(targets, dim=0))

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(input.detach(), 1)
        loss = criterion(outputs, label.detach())
        loss.backward()
        optimizer.step()

        mean_loss = 0.99*mean_loss + 0.01*loss.data[0]
        pbar.set_description('Epoch [%d/%d] Loss : %f' % (epoch+1, num_epochs, mean_loss))
    pbar.close()

    pbar = tqdm(range(len(sst)//64))
    random.shuffle(sst)
    for i in pbar:
        lists = sst[i*64:(i+1)*64]
        batch = []
        targets = []
        for words, target in lists:
            batch.append(torch.LongTensor(words))
            targets.append(torch.FloatTensor(target))

        input = Variable(torch.stack(batch, dim=0))
        label = Variable(torch.stack(targets, dim=0))

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(input.detach(), 2)
        loss = criterion(outputs, label.detach())
        loss.backward()
        optimizer.step()

        pbar.set_description('Epoch [%d/%d] Loss : %f' % (epoch+1, num_epochs, loss.data[0]))
    pbar.close()

    avg_loss = 0
    cnn.eval()
    for i in range(len(testset)//64):
        lists = testset[i*64:(i+1)*64]
        batch = []
        targets = []
        for words, target in lists:
            batch.append(torch.LongTensor(words))
            targets.append(torch.FloatTensor(target))

        input = Variable(torch.stack(batch, dim=0))
        label = Variable(torch.stack(targets, dim=0))

        # Forward
        outputs = cnn(input.detach(),1)
        loss = criterion(outputs, label.detach())

        avg_loss += loss.data[0]

    cnn.train()

    avg_loss /= (len(testset)//64)
    print("val loss : %f" % (avg_loss))

    # Save the Trained Model
    torch.save(cnn.state_dict(), 'cnn_epoch%d.pkl'%(epoch+1))
    f = open("loss.txt", 'a')
    f.write('Epoch [%d/%d] Loss : %f\n' % (epoch+1, num_epochs, avg_loss))
    f.close()
