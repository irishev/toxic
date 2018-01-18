import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transf
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm

# Hyper Parameters
sequence_length = 28
input_size = 100
hidden_size = 300
num_layers = 2
num_classes = 10
batch_size = 64
num_epochs = 30
learning_rate = 0.0005

# MNIST Dataset
train_dataset = dsets.CIFAR10(root='./data/',
                            train=True,
                            transform=transf.Compose([
                                transf.RandomHorizontalFlip(),
                                transf.RandomRotation(180),
                                transf.ColorJitter(0.3, 0.3, 0.3, 0.3),
                                transf.ToTensor(),
                                transf.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ]),
                            download=True)

test_dataset = dsets.CIFAR10(root='./data/',
                           train=False,
                           transform=transf.Compose([
                            transf.ToTensor(),
                            transf.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ]))

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class Convnet(nn.Module):
    def __init__(self):
        super(Convnet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
        )
        self.pool1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
        )
        self.pool2 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(8192, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        y = torch.prod(x, 1)
        cov1 = torch.sum(y)
        x = self.pool1(x)
        x = self.layer2(x)
        y = torch.prod(x, 1)
        cov2 = torch.sum(y)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, cov1, cov2

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, kernel_size=kernel_size, padding=padding, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Conv3x3(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, **kwargs):
        super(Conv3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, kernel_size=(kernel_size, 1), padding=(padding, 0), **kwargs),
            nn.Conv2d(out_channels, out_channels, bias=False, kernel_size=(1, kernel_size), padding=(0, padding), groups=out_channels)
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class ResNeXtModule(nn.Module):
    def __init__(self, c, groups, cgroups=1):
        super(ResNeXtModule, self).__init__()
        self.l1 = nn.Sequential(
            BasicConv2d(c, c//2, groups=cgroups, kernel_size=1),
            Conv3x3(c//2, c//2, groups=groups, padding=1),
            BasicConv2d(c//2, c, groups=cgroups, kernel_size=1),
        )

    def forward(self, x):
        return self.l1(x)+x

class Upsample(nn.Module):
    def __init__(self,in_c, groups=1):
        super(Upsample,self).__init__()
        self.l1 = BasicConv2d(in_c, in_c, kernel_size=3, stride=2, padding=1, groups=groups)

    def forward(self, x):
        x = [self.l1(x), F.max_pool2d(x, kernel_size=3, stride=2, padding=1)]
        return torch.cat(x, dim=1)

class ResNeXt(nn.Module):
    def __init__(self):
        super(ResNeXt,self).__init__()
        self.l1 = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, padding=1),
            BasicConv2d(32, 64, kernel_size=3, padding=1),
            Upsample(64)
        )
        self.l2 = nn.Sequential(
            ResNeXtModule(128, 32),
            ResNeXtModule(128, 32),
            Upsample(128)
        )
        self.l3 = nn.Sequential(
            ResNeXtModule(256, 32),
            ResNeXtModule(256, 32),
            ResNeXtModule(256, 32),
            Upsample(256)
        )
        self.l4 = nn.Sequential(
            ResNeXtModule(512, 32),
            ResNeXtModule(512, 32),
            ResNeXtModule(512, 32),
            ResNeXtModule(512, 32),
            ResNeXtModule(512, 32),
            Upsample(512)
        )
        self.l5 = nn.Sequential(
            ResNeXtModule(1024, 32),
            ResNeXtModule(1024, 32),
            ResNeXtModule(1024, 32),
            Upsample(1024)
        )
        self.fc = nn.Linear(2048,10)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0),-1)
        x = F.dropout(x, training=self.training)
        x = self.fc(x)

        return x

class ModuleA(nn.Module):
    def __init__(self):
        super(ModuleA, self).__init__()
        self.l1 = nn.Sequential(
            BasicConv2d(3, 128, kernel_size=3, padding=1),
            #BasicConv2d(32, 64, kernel_size=5),
            #Upsample(64, 128)
        )
        self.l2 = nn.Sequential(
            ResNeXtModule(128, 32),
            ResNeXtModule(128, 32),
            ResNeXtModule(128, 32),
            #Upsample(128, 128)
        )
        self.l3 = nn.Sequential(
            ResNeXtModule(128, 32),
            ResNeXtModule(128, 32),
            ResNeXtModule(128, 32),
            ResNeXtModule(128, 32),
            ResNeXtModule(128, 32),
            Upsample(128, 384)
        )
        self.l4 = nn.Sequential(
            ResNeXtModule(512, 32),
            ResNeXtModule(512, 32),
            ResNeXtModule(512, 32),
            ResNeXtModule(512, 32),
            ResNeXtModule(512, 32),
            ResNeXtModule(512, 32),
            ResNeXtModule(512, 32),
            #Upsample(512, 512)
        )
        self.l5 = nn.Sequential(
            ResNeXtModule(512, 32),
            ResNeXtModule(512, 32),
            Upsample(512,2048)
        )
        self.fc = nn.Linear(2048,10)

    def forward(self, x):
        x = self.l1(x)
        out = self.l2(x)
        #x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        #x = torch.cat([x, out], 1)
        out = self.l3(out)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = torch.cat([x, out], 1)
        out = self.l4(x)
        #x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        #x = torch.cat([x, out], 1)
        x = self.l5(out)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0),-1)
        x = F.dropout(x, training=self.training)
        x = self.fc(x)

        return x

rnn = ResNeXt()
#rnn.load_state_dict(torch.load('rnn.pkl'))
rnn.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)

# Train the Model
for epoch in range(num_epochs):
    scheduler.step()
    pbar = tqdm(total=len(train_loader))
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = rnn(images)
        loss = criterion(outputs, labels)
        pbar.set_description("loss : %f" % loss.data[0])
        loss.backward()
        optimizer.step()
        pbar.update()

    # Test the Model
    correct = 0
    total = 0
    rnn.eval()
    for images, labels in test_loader:
        images = Variable(images).cuda()
        outputs = rnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
    rnn.train()
    print('Test Accuracy of the model on the 10000 test images: %f %%' % (100.0 * correct / total))
    pbar.close()


# Save the Model
torch.save(rnn.state_dict(), 'rnn.pkl')