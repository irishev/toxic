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

f = open('C:/Users/dilab3/Downloads/train.csv/train.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)

dataset = []
length = []
for i, line in enumerate(rdr):
    if i!=0:
        l = line[1]
        chars = [ord(c) for c in l]
        for i, c in enumerate(chars):
            if c>128:
                chars[i]=0
        length.append(len(chars))
        if len(chars)<512:
            for i in range(0,512-len(chars)):
                chars.append(0)
        else:
            chars = chars[0:512]

        target = line[2:]
        target = list(map(float, target))

        dataset.append((chars,target))

length.sort()


f.close()


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

embedding = nn.Embedding(128, 64, padding_idx=0)

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.embed = embedding
        self.conv1 = nn.Sequential(
            Conv(1, 16, kernel_size=(1, 64)),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            Conv(1, 16, kernel_size=(3, 64), padding=(1, 0)),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            Conv(1, 16, kernel_size=(5, 64), padding=(2, 0)),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            Conv(48, 1, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(512, 128),
            nn.LogSoftmax(dim=1)
        )

        self.wconv = nn.Sequential(
            Conv(1, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            Conv(128, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1)
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64, 6)
        )

    def forward(self, x, mode):
        for i in range(0,64):
            for j in range(0,512):
                if(x.data[i][j]>128):
                    print(x.data[i][j])
        x = self.embed(x).unsqueeze(1)
        x = [self.conv1(x), self.conv3(x), self.conv5(x)]
        x = torch.cat(x, 1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.view(x.size(0), 1, -1, 1)
        x = self.wconv(x)
        x = x.view(x.size(0), -1)
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

        # mean_loss = 0.99*mean_loss + 0.01*loss.data[0]
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

