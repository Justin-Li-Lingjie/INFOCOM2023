import torch
import torch.nn as nn
import numpy as np

import data_sets
import defences
import user
import os
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
import threading
# import tensorflow as tf




class Server:
    def __init__(self, users, malicious_proportion, batch_size, learning_rate, fading_rate, momentum, data_set):

        self.users = users
        self.mal_prop = malicious_proportion
        self.learning_rate = learning_rate
        self.fading_rate = fading_rate
        self.momentum = momentum
        self.data_set = data_set
        if data_set == data_sets.MNIST:
            self.test_net = data_sets.MnistNet()
        elif data_set == data_sets.CIFAR10:
            self.test_net = data_sets.Cifar10Net()
        elif data_set == data_sets.FAMNIST:
            self.test_net = data_sets.FaMnistNet()
        else:
            raise Exception("Unknown dataset {}".format(data_set))
        self.criterion = nn.CrossEntropyLoss()
        self.test_loader = torch.utils.data.DataLoader(self.test_net.dataset(False), batch_size=batch_size, shuffle=False)
        self.test_net.to(device)
        self.current_weights = np.concatenate([i.data.cpu().numpy().flatten() for i in self.test_net.parameters()])
        self.users_grads = np.empty((len(users), len(self.current_weights)), dtype=self.current_weights.dtype)
        self.velocity = np.zeros(self.current_weights.shape, self.users_grads.dtype)
        self.Meta = None
        # score


    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        directory = "runs/%s/" % (self.data_set)
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + filename
        torch.save(state, filename)

        # shutil.copyfile(filename, 'runs/%s/' % self.data_set + 'model_best.pth.tar')

    def calc_learning_rate(self, cur_epoch):
        lr = self.learning_rate * self.fading_rate / (cur_epoch + self.fading_rate)
        return lr

    def dispatch_weights(self, cur_epoch):
        for usr in self.users:
            # usr.step(self.current_weights, self.calc_learning_rate(cur_epoch))
            usr.step(self.current_weights, 0.01)


    def get_MetaData(self):
        return self.Meta

    # collect Meta Data from users
    def collect_MetaData(self, users):
        OriginMetaX, OriginMetaY = users[0].MetaX, users[0].MetaY
        i = 1
        Limit = len(users)
        while i < Limit:
            TempMetaX, TempMetaY = users[i].MetaX, users[i].MetaY
            # TempMeta = [TempMetaX, TempMetaY]


            # Meta = tf.concat([Meta, TempMeta], 1)
            OriginMetaX = torch.cat([OriginMetaX, TempMetaX], 0)
            OriginMetaY = torch.cat([OriginMetaY, TempMetaY], 0)

            i = i + 1
        self.Meta = (OriginMetaX, OriginMetaY)


    # get the updated weights from users
    def collect_gradients(self):
        # no_mal_users = [u for u in self.users if u.is_malicious == 0]
        # self.users_grads = np.empty((len(no_mal_users), len(self.current_weights)), dtype=self.current_weights.dtype)
        #
        # for idx, usr in enumerate(no_mal_users):
        #     self.users_grads[idx, :] = usr.grads
        for idx, usr in enumerate(self.users):
            self.users_grads[idx, :] = usr.grads
    # defend against malicious users
    def defend(self, defence_method, cur_epoch):
        current_grads = defences.defend[defence_method](self.users_grads, len(self.users), int(len(self.users)*self.mal_prop), self.users)
        # current_grads = defences.defend[defence_method](self.users_grads, len(self.users_grads), 0)

        # self.velocity = self.momentum * self.velocity - self.learning_rate * current_grads
        #
        # self.current_weights += self.velocity
        self.current_weights = current_grads

    def test(self):
        user.row_into_parameters(self.current_weights, self.test_net.parameters())
        test_loss = 0
        correct = 0
        # self.test_net.eval()
        self.test_loader = torch.utils.data.DataLoader(self.test_net.dataset(False), batch_size=100, shuffle=False, num_workers=5)
        with torch.no_grad():
            for data, target in self.test_loader:
                if self.data_set == data_sets.MNIST:
                    data = data.view(-1, 28 * 28)
                data = data.to(device)
                target = target.to(device)
                net_out = self.test_net(data)
                loss = self.criterion(net_out, target)
                # sum up batch loss
                test_loss += loss.data.item()*target.size(0)
                pred = torch.max(net_out, 1)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data).sum()
        test_loss /= len(self.test_loader.dataset)


        return test_loss, correct












