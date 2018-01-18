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

f = open('C:/test/train.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)

dataset = []
length = []
for i, line in enumerate(rdr):
    if i!=0:
        l = re.sub("[^a-z \',.?!]", "",  str.lower(line[1]))
        l = re.sub(r'([a-z \',.?!])\1{2,}', r'\1\1', l)
        chars = [ord(c) for c in l]
        for i, c in enumerate(chars):
            if c==39:
                chars[i]=27
            elif c==33:
                chars[i]=28
            elif c==44:
                chars[i]=29
            elif c==46:
                chars[i]=30
            elif c==63:
                chars[i]=31
            elif c==32:
                chars[i]=32
            elif (c>122) or (c<97):
                chars[i]=0
            else:
                chars[i] -= 96
        length.append(len(chars))
        if len(chars)<4096:
            for i in range(0,4096-len(chars)):
                chars.append(0)
        else:
            chars = chars[0:4096]

        target = line[2:]
        target = list(map(float, target))

        dataset.append((chars,target))

length.sort()
print(length[-1])


f.close()


class Conv(nn.Module):
    def __init__(self, in_c, out_c, **kwargs):
        super(Conv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, bias=True, **kwargs),
            # nn.BatchNorm2d(out_c),
        )
    def forward(self, x):
        return self.layer1(x)

class DCNModule(nn.Module):
    def __init__(self, ch, mid_ch, c):
        super(DCNModule,self).__init__()
        self.eql1 = Conv(ch, mid_ch, kernel_size=1)
        self.layer1 = Conv(mid_ch, mid_ch, kernel_size=(3, 1), padding=(1, 0), groups=c)
        self.layer2 = Conv(mid_ch, mid_ch, kernel_size=(5, 1), padding=(2, 0), groups=c)
        self.layer3 = Conv(mid_ch, mid_ch, kernel_size=(7, 1), padding=(3, 0), groups=c)
        self.eql2 = Conv(mid_ch * 3, ch, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.eql1(x), inplace=True)
        out = [self.layer1(out), self.layer2(out), self.layer3(out)]
        out = F.relu(torch.cat(out, 1), inplace=True)
        return F.relu(self.eql2(out)+x, inplace=True)

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


# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            Conv(1, 64, kernel_size=(3, 32), padding=(1, 0)),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            Conv(1, 64, kernel_size=(5, 32), padding=(2, 0)),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            Conv(1, 128, kernel_size=(7, 32), padding=(3, 0)),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            DCNModule(256, 64, 16),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            DCNModule(256, 64, 16),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            DCNModule(256, 64, 16),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            DCNModule(256, 64, 16),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            DCNModule(256, 64, 16),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            DCNModule(256, 64, 16),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            DCNModule(256, 64, 16),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            DCNModule(256, 64, 16),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            DCNModule(256, 64, 16),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            DCNModule(256, 64, 16),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            DCNModule(256, 64, 16),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            DCNModule(256, 64, 16),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        '''self.fc1 = nn.Sequential(
            # nn.Dropout(),
            nn.LogSoftmax(dim=1)
        )

        self.wconv = nn.Sequential(
            Conv(1, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            Conv(128, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1)
        )'''

        self.fc2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = [self.conv1(x), self.conv3(x), self.conv5(x)]
        x = torch.cat(x, 1)
        x = self.conv(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        #x = x.view(x.size(0), 1, -1, 1)
        #x = self.wconv(x)
        x = x.view(x.size(0), -1)
        x = self.fc2(x)
        return x

cnn = CNN().cuda()

learning_rate = 0.001
num_epochs = 20

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
open("loss.txt", 'w')
mean_loss = 0.2


random.shuffle(dataset)
trainset = dataset[:len(dataset)*9//10]
testset = dataset[len(dataset)*9//10:]
# Train the Model
for epoch in range(num_epochs):
    # scheduler.step()
    pbar = tqdm(range(len(trainset)//64))
    random.shuffle(trainset)
    for i in pbar:
        lists = trainset[i*64:(i+1)*64]
        batch = []
        targets = []
        for words, target in lists:
            indices = torch.LongTensor(words).view(-1, 1)
            one_hot = torch.zeros(4096, 33)
            one_hot.scatter_(1, indices, 1)
            batch.append(one_hot.narrow(1,1,32))
            targets.append(torch.FloatTensor(target))

        input = torch.stack(batch, dim=0)
        targets = torch.stack(targets, dim=0)

        input = Variable(input.cuda())
        label = Variable(targets.cuda())

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(input)
        loss = criterion(outputs, label)
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
            indices = torch.LongTensor(words).view(-1, 1)
            one_hot = torch.zeros(4096, 33)
            one_hot.scatter_(1, indices, 1)
            batch.append(one_hot.narrow(1, 1, 32))
            targets.append(torch.FloatTensor(target))

        input = Variable(torch.stack(batch, dim=0)).cuda()
        label = Variable(torch.stack(targets, dim=0)).cuda()

        # Forward
        outputs = cnn(input)
        loss = criterion(outputs, label)

        avg_loss += loss.data[0]

    cnn.train()

    avg_loss /= (len(testset)//64)
    print("val loss : %f" % (avg_loss))

    # Save the Trained Model
    torch.save(cnn.state_dict(), 'cnn_epoch%d.pkl'%(epoch+1))
    f = open("loss.txt", 'a')
    f.write('Epoch [%d/%d] Loss : %f\n' % (epoch+1, num_epochs, avg_loss))
    f.close()

