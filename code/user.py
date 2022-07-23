import functools
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import data_sets
from sklearn.model_selection import train_test_split
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def flatten_params(params):
    return np.concatenate([i.data.cpu().numpy().flatten() for i in params])


def row_into_parameters(row, parameters):
    offset = 0
    for param in parameters:
        new_size = functools.reduce(lambda x,y:x*y, param.shape)
        current_data = row[offset:offset + new_size]

        param.data[:] = torch.from_numpy(current_data.reshape(param.shape))
        offset += new_size


class User:

    def __init__(self, user_id, batch_size, is_malicious, users_count, momentum, probability ,credit_id, score = 0, data_set=data_sets.MNIST):
        self.is_malicious = is_malicious
        self.credit_id = credit_id
        self.user_id = user_id

        self.learning_rate = None

        self.data_set = data_set
        self.momentum = momentum
        self.grads = None
        self.probability = probability
        # score
        self.score = score
        if data_set == data_sets.MNIST:
            self.net = data_sets.MnistNet()
        elif data_set == data_sets.CIFAR10:
            self.net = data_sets.Cifar10Net()
        elif data_set == data_sets.FAMNIST:
            self.net = data_sets.FaMnistNet()
        self.criterion = nn.CrossEntropyLoss()
        self.original_params = None
        self.net.to(device)
        self.score = score
        dataset = self.net.dataset(True)
        sampler = None
        # if users_count > 1:
        #     sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=users_count, rank=user_id)
        self.train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=100, shuffle=True)
        self.train_iterator = iter(cycle(self.train_loader))
        # self.MetaX, self.MetaY = self.getMetaData()
        # self.MetaX = self.MetaX.tolist()
        # self.MetaY = self.MetaY.tolist()



    def train(self, data, target):
        if self.data_set == data_sets.MNIST :
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            data = data.view(-1, 28 * 28)
        data = data.to(device)
        target = target.to(device)
        self.optimizer.zero_grad()
        net_out = self.net(data)
        # for flip 标签翻转攻击
        # if self.is_malicious == 1:
        #     for i in range(0,len(target)):
        #         if target[i] == 1 or target[i] == 2 or target[i] == 3 or target[i] == 4 or target[i] == 5 or target[i] == 6:
        #             target[i] = 0

        loss = self.criterion(net_out, target)
        loss.backward()
        self.optimizer.step() # not stepping because reporting the gradients and the server is performing the step

    # user gets initial weights and learn the new gradients based on its data
    def step(self, current_params, learning_rate):
        row_into_parameters(current_params, self.net.parameters())
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)
        self.net.train()
        for j in range(0, 1):
            for i, data in enumerate(self.train_loader, 0):
                img, label = data
                self.train(img, label)
        self.grads = np.concatenate([param.data.cpu().numpy().flatten() for param in self.net.parameters()])
    def test(self):
        test_loss = 0
        correct = 0
        self.net.eval()
        self.test_loader = torch.utils.data.DataLoader(self.net.dataset(False), batch_size=100, shuffle=False, num_workers=5)
        with torch.no_grad():
            for data, target in self.test_loader:
                if self.data_set == data_sets.MNIST:
                    data = data.view(-1, 28 * 28)
                data = data.to(device)
                target = target.to(device)
                net_out = self.net(data)
                loss = self.criterion(net_out, target)
                # sum up batch loss
                test_loss += loss.data.item()
                pred = torch.max(net_out, 1)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data).sum()
        print(correct/len(self.test_loader.dataset))

if __name__ == '__main__':
    user_id = 1
    batch_size = 83
    is_mal = False
    users_count = 10
    momentum = 0.9
    user = User(user_id, batch_size, is_mal, users_count, momentum)
    # print (user.train_loader)
    # data, target = next(self.train_iterator)
    print (len(user.train_loader))








