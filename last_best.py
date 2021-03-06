import gensim
import csv
import re
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm

# Load Google's pre-trained Word2Vec model.
print('Loading word2vec...')
model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/dilab3/Downloads/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True)
print('Done!')

f = open('C:/Users/dilab3/Downloads/train.csv/train.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)

dataset = []
for i, line in enumerate(rdr):
    if i!=0:
        words = re.sub("[^\w]", " ",  str.lower(line[1])).split()
        wordlist = []
        for word in words:
            if word in model.vocab:
                wordlist.append(word)
        if len(wordlist) < 16:
            for _ in range(16-len(wordlist)):
                wordlist.append('')
        elif len(wordlist) > 16:
            wordlist = wordlist[0:16]


        target = line[2:]
        target = list(map(float, target))

        dataset.append((wordlist,torch.FloatTensor(target)))

f.close()

class Conv(nn.Module):
    def __init__(self, in_c, out_c, **kwargs):
        super(Conv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, **kwargs),
            # nn.BatchNorm2d(out_c),
        )
    def forward(self, x):
        return self.layer1(x)

class DCNModule(nn.Module):
    def __init__(self, in_c, out_c, c):
        super(DCNModule,self).__init__()
        self.layer1 = nn.Sequential(
            Conv(in_c, in_c//2, kernel_size=1),
            nn.ReLU(),
            Conv(in_c//2, in_c//2, kernel_size=(3,1), padding=(1,0), groups=c),
            nn.ReLU(),
            Conv(in_c//2, out_c, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer1(x)+x

class Downsample(nn.Module):
    def __init__(self, in_c, out_c, c):
        super(Downsample,self).__init__()
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=2, padding=(1, 0)),
            Conv(in_c, out_c, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer1(x)


# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            Conv(1, 300, kernel_size=(3, 300), padding=(1, 0)),
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
        return x

cnn = CNN()

learning_rate = 0.001
num_epochs = 20

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=1e-5)
open("loss.txt", 'w')
mean_loss = 3.

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
            vecs = []
            for word in words:
                if word=='':
                    vecs.append(torch.zeros(300))
                else:
                    vecs.append(torch.Tensor(model.wv[word]))
            batch.append(torch.unsqueeze(torch.stack(vecs, dim=0), dim=0))
            targets.append(target)
        input = torch.stack(batch, dim=0)
        targets = torch.stack(targets, dim=0)
        input = Variable(input)
        label = Variable(targets)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(input)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        mean_loss = 0.9*mean_loss + 0.1*loss.data[0]
        pbar.set_description('Epoch [%d/%d] Loss : %f' % (epoch+1, num_epochs, mean_loss))
    pbar.close()

    avg_loss = 0
    for i in range(len(testset)//64):
        lists = testset[i*64:(i+1)*64]
        batch = []
        targets = []
        for words, target in lists:
            vecs = []
            for word in words:
                if word=='':
                    vecs.append(torch.zeros(300))
                else:
                    vecs.append(torch.Tensor(model.wv[word]))
            batch.append(torch.unsqueeze(torch.stack(vecs, dim=0), dim=0))
            targets.append(target)
        input = torch.stack(batch, dim=0)
        targets = torch.stack(targets, dim=0)
        input = Variable(input)
        label = Variable(targets)

        # Forward
        outputs = cnn(input)
        loss = criterion(outputs, label)

        avg_loss += loss.data[0]

    avg_loss /= (len(testset)//64)
    print("val loss : %f" % (avg_loss))

    # Save the Trained Model
    torch.save(cnn.state_dict(), 'cnn_epoch%d.pkl'%(epoch+1))
    f = open("loss.txt", 'a')
    f.write('Epoch [%d/%d] Loss : %f\n' % (epoch+1, num_epochs, avg_loss))
    f.close()
