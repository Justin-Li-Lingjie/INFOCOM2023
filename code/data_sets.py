import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import math
from collections import OrderedDict


MNIST = 'MNIST'
CIFAR10 = 'CIFAR10'
CIFAR100 = 'CIFAR100'
FAMNIST = 'FAMNIST'


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def dataset(self, is_train, transform=None):
        t = [transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
        if transform:
            t.append(transform)
        return datasets.MNIST('./mnist_data', download=False, train=is_train, transform=transforms.Compose(t))


class FaMnistNet(nn.Module):
    def __init__(self):
        super(FaMnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)  # 1表示输入通道，20表示输出通道，5表示conv核大小，1表示conv步长
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


    def dataset(self, is_train, transform=None):
        t = [transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
        if transform:
            t.append(transform)
        return datasets.FashionMNIST('./famnist_data', download=False, train=is_train, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))


class CNet(nn.Module):
    def __init__(self):
        super(CNet, self).__init__()

        self.cnn = nn.Sequential(
            # 卷积层1，3通道输入，96个卷积核，核大小7*7，步长2，填充2
            # 经过该层图像大小变为32-7+2*2 / 2 +1，15*15
            # 经3*3最大池化，2步长，图像变为15-3 / 2 + 1， 7*7
            nn.Conv2d(3, 96, 7, 2, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0),

            # 卷积层2，96输入通道，256个卷积核，核大小5*5，步长1，填充2
            # 经过该层图像变为7-5+2*2 / 1 + 1，7*7
            # 经3*3最大池化，2步长，图像变为7-3 / 2 + 1， 3*3
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0),

            # 卷积层3，256输入通道，384个卷积核，核大小3*3，步长1，填充1
            # 经过该层图像变为3-3+2*1 / 1 + 1，3*3
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(inplace=True),

            # 卷积层3，384输入通道，384个卷积核，核大小3*3，步长1，填充1
            # 经过该层图像变为3-3+2*1 / 1 + 1，3*3
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(inplace=True),

            # 卷积层3，384输入通道，256个卷积核，核大小3*3，步长1，填充1
            # 经过该层图像变为3-3+2*1 / 1 + 1，3*3
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            # 256个feature，每个feature 3*3
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.cnn(x)

        # x.size()[0]: batch size
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x
    @staticmethod
    def dataset(is_train, transform=None):
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        return datasets.CIFAR10(root='./cifar10_data', download=False, train=is_train,
                                       transform=t)


class Cifar10Net(nn.Module):
    def __init__(self):
        super(Cifar10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(16, 64, 4)
        self.pool2 = nn.MaxPool2d(4)
        self.fc1 = nn.Linear(64 * 1 * 1, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

    @staticmethod
    def dataset(is_train, transform=None):
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        return datasets.CIFAR10(root='./cifar10_data', download=False, train=is_train,
                                       transform=t)


class AlexNet(nn.Module):
    def __init__(self,num_inputs,num_classes):
        super(AlexNet,self).__init__()
        self.net1=nn.Sequential(OrderedDict([
            ('conv1',nn.Conv2d(num_inputs,96,kernel_size=11,stride=4,padding=2)),   #卷积层1
            ('relu1',nn.ReLU(inplace=True)), #直接计算，不拷贝，节省内存与时间
            ('LRN1',nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2)),     #LRN层1
            ('pool1',nn.MaxPool2d(kernel_size=3,stride=2)),                        #池化层1
            ('conv2',nn.Conv2d(96,256,kernel_size=5,padding=2)),                  #卷积层2
            ('relu2',nn.ReLU(inplace=True)),
            ('LRN2',nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)),  #LRN层2
            ('pool2',nn.MaxPool2d(kernel_size=3,stride=2)),                      #池化层2
            ('conv3',nn.Conv2d(256,384,kernel_size=3,padding=1)),                  #卷积层3
            ('relu3',nn.ReLU(inplace=True)),
            ('conv4',nn.Conv2d(384,384,kernel_size=3,padding=1)),                 #卷积层4
            ('relu4',nn.ReLU(inplace=True)),
            ('conv5',nn.Conv2d(384,256,kernel_size=3,padding=1)),                  #卷积层5
            ('relu5',nn.ReLU(inplace=True)),
            ('pool3',nn.MaxPool2d(kernel_size=3,stride=2))                        #池化层3
        ]))
        self.net2=nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216, 4096),                      #全连接层1 dropout
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),                      #全连接层2 dropout
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)              #全连接层3
        )
    def forward(self,x):
        #print(x.size())
        out=self.net1(x)
        #print(out.size())
        out=out.view(-1,9216)
        out=self.net2(out)
        return out
    def dataset(is_train, transform=None):
        return datasets.CIFAR10(root='./cifar10_data', download=False, train=is_train,
                                       transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
# sklearn.model_selection.train_test_split
import numpy as np

'''
残差块
in_channels, out_channels：残差块的输入、输出通道数
对第一层，in out channel都是64，其他层则不同
对每一层，如果in out channel不同， stride是1，其他层则为2
'''
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        # 残差块的第一个卷积
        # 通道数变换in->out，每一层（除第一层外）的第一个block
        # 图片尺寸变换：stride=2时，w-3+2 / 2 + 1 = w/2，w/2 * w/2
        # stride=1时尺寸不变，w-3+2 / 1 + 1 = w
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 残差块的第二个卷积
        # 通道数、图片尺寸均不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差块的shortcut
        # 如果残差块的输入输出通道数不同，则需要变换通道数及图片尺寸，以和residual部分相加
        # 输出：通道数*2 图片尺寸/2
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # 通道数相同，无需做变换，在forward中identity = x
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


'''
定义网络结构
'''
class ResNet34(nn.Module):
    def __init__(self, block):
        super(ResNet34, self).__init__()

        # 初始卷积层核池化层
        self.first = nn.Sequential(
            # 卷基层1：7*7kernel，2stride，3padding，outmap：32-7+2*3 / 2 + 1，16*16
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 最大池化，3*3kernel，1stride（32的原始输入图片较小，不再缩小尺寸），1padding，
            # outmap：16-3+2*1 / 1 + 1，16*16
            nn.MaxPool2d(3, 1, 1)
        )

        # 第一层，通道数不变
        self.layer1 = self.make_layer(block, 64, 64, 3, 1)

        # 第2、3、4层，通道数*2，图片尺寸/2
        self.layer2 = self.make_layer(block, 64, 128, 4, 2)  # 输出8*8
        self.layer3 = self.make_layer(block, 128, 256, 6, 2)  # 输出4*4
        self.layer4 = self.make_layer(block, 256, 512, 3, 2)  # 输出2*2

        self.avg_pool = nn.AvgPool2d(2)  # 输出512*1
        self.fc = nn.Linear(512, 10)

    def make_layer(self, block, in_channels, out_channels, block_num, stride):
        layers = []

        # 每一层的第一个block，通道数可能不同
        layers.append(block(in_channels, out_channels, stride))

        # 每一层的其他block，通道数不变，图片尺寸不变
        for i in range(block_num - 1):
            layers.append(block(out_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)

        # x.size()[0]: batch size
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x

    @staticmethod
    def dataset(is_train, transform=None):
        t = [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        if transform:
            t.append(transform)
        return datasets.CIFAR10(root='./cifar10_data', download=False, train=is_train,
                                       transform=transforms.Compose(t))

if __name__ == '__main__':
    net = MnistNet()
    dataset = net.dataset(True)
    # sampler = None
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=users_count, rank=user_id)
    users_count = 10
    batch_size = 83
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=10, rank=1)

    train_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size, shuffle=sampler is None)
    train_iterator = iter(cycle(train_loader))
    X, y = next(train_iterator)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.11, random_state = 42, stratify = y)


    print (X_test, y_test)
    print (len(X), len(y))
    print (len(train_loader))














