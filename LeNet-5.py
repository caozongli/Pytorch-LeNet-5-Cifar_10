#                        .::::.
#                      .::::::::.
#                     :::::::::::
#                  ..:::::::::::'
#               '::::::::::::'
#                 .::::::::::
#            '::::::::::::::..
#                 ..::::::::::::.
#               ``::::::::::::::::
#                ::::``:::::::::'        .:::.
#               ::::'   ':::::'       .::::::::.
#             .::::'      ::::     .:::::::'::::.
#            .:::'       :::::  .:::::::::' ':::::.
#           .::'        :::::.:::::::::'      ':::::.
#          .::'         ::::::::::::::'         ``::::.
#      ...:::           ::::::::::::'              ``::.
#     ```` ':.          ':::::::::'                  ::::..
#                        '.:::::'                    ':'````..

import os
import numpy as np

import torch
from torch.nn import init
import torch.nn.modules as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader, Sampler
import pickle

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# from sklearn.metrics import accuracy_score


plt.rcParams['figure.figsize'] = (20, 16)
plt.rcParams['image.cmap'] = 'gray'

num_epoch = 100
batch = 128
num_classes = 10


def show_images(images):
    images_shape = images.shape
    images = images.reshape(images.shape[0], -1)
    sqrt = int(np.ceil(np.sqrt(images.shape[0])))
    # sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))
    fig = plt.figure(figsize=(sqrt, sqrt))
    gs = gridspec.GridSpec(sqrt, sqrt)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        if np.size(images_shape) < 3:
            plt.imshow(img.reshape(images_shape[1], images_shape[2]))
        else:
            plt.imshow(img.reshape(images_shape[1], images_shape[2], images_shape[3]))

    return


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(6)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.conv3_3 = nn.Conv2d(3, 1, kernel_size=5, stride=1, bias=True)
        self.conv3_4 = nn.Conv2d(4, 1, kernel_size=5, stride=1, bias=True)
        self.conv3_6 = nn.Conv2d(6, 1, kernel_size=5, stride=1, bias=True)
        self.bn3 = nn.BatchNorm2d(16)

        self.maxpool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(16, 120, kernel_size=5, bias=True)
        self.bn5 = nn.BatchNorm2d(120)

        self.full6 = nn.Linear(120, 84)
        self.full7 = nn.Linear(84, 10)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool2(x)

        x_data = []
        x_n = x.data.cpu().numpy()
        x_data.append(x[:, 0:3:, :, :])
        x_data.append(x[:, 1:4:, :, :])
        x_data.append(x[:, 2:5:, :, :])
        x_data.append(x[:, 3:6:, :, :])

        x_data.append(torch.from_numpy(np.concatenate((x_n[:, 0, :, :][:, np.newaxis, :, :], x_n[:, 4:6, :, :]), axis=1)))
        x_data.append(torch.from_numpy(np.concatenate((x_n[:, 0:2:, :, :], x_n[:, 5, :, :][:, np.newaxis, :, :]), axis=1)))

        x_data.append(x[:, 0:4:, :, :])
        x_data.append(x[:, 1:5:, :, :])
        x_data.append(x[:, 2:6:, :, :])
        x_data.append(torch.from_numpy(np.concatenate((x_n[:, 0, :, :][:, np.newaxis, :, :], x_n[:, 3:6, :, :]), axis=1)))
        x_data.append(torch.from_numpy(np.concatenate((x_n[:, 0:2:, :, :], x_n[:, 4:6, :, :]), axis=1)))
        x_data.append(torch.from_numpy(np.concatenate((x_n[:, 0:3:, :, :], x_n[:, 5, :, :][:, np.newaxis, :, :]), axis=1)))

        x_data.append(torch.from_numpy(np.concatenate((x_n[:, 0:2:, :, :], x_n[:, 3:5:, :, :]), axis=1)))
        x_data.append(torch.from_numpy(np.concatenate((x_n[:, 1:3:, :, :], x_n[:, 4:6:, :, :]), axis=1)))
        x_data.append(torch.from_numpy(np.concatenate((x_n[:, 0, :, :][:, np.newaxis, :, :], x_n[:, 2:4:, :, :], x_n[:, 5, :, :][:, np.newaxis, :, :]), axis=1)))
        x_data.append(x)
        x_data = np.array(x_data)

        out = self.conv3_3(Variable(x_data[0]).float().cuda()).data.cpu().numpy()
        for n in range(1, 16):
            if n < 6:
                out = np.concatenate((out, self.conv3_3(Variable(x_data[n]).float().cuda()).data.cpu().numpy()), axis=1)
            elif 6 <= n < 15:
                out = np.concatenate((out, self.conv3_4(Variable(x_data[n]).float().cuda()).data.cpu().numpy()), axis=1)
            else:
                out = np.concatenate((out, self.conv3_6(Variable(x_data[n]).float().cuda()).data.cpu().numpy()), axis=1)

        out = Variable(torch.from_numpy(out)).float().cuda()
        out = self.bn3(out)
        out = self.maxpool4(out)
        out = self.conv5(out)
        out = out.view(x.size(0), -1)
        out = self.full6(out)
        out = self.full7(out)
        out = self.sigmoid(out)
        return out


path = os.path.abspath(os.path.join(os.getcwd(), "/caoshujian/Dataset_myself/cifar_10/test_batch"))
# path = 'G:/Study/Deep_Learning_cs231n/cifar-10-python/cifar-10-batches-py/test_batch'
test_img = None
test_labs = None
with open(path, 'rb') as lbpath:
    dict = pickle.load(lbpath, encoding='bytes')
    test_img = dict[b'data'].reshape(dict[b'data'].shape[0], 3, 32, 32)
    test_labs = dict[b'labels']
test_img = test_img / 255.0
test_img = test_img.astype(np.float32)

images = []
labels = []
path = '/caoshujian/Dataset_myself/cifar_10'
# path = 'G:/Study/Deep_Learning_cs231n/cifar-10-python/cifar-10-batches-py'
for i in range(1, 6):
    f = os.path.join(path, 'data_batch_%d' % i)
    with open(f, 'rb') as lbpath:
        dict = pickle.load(lbpath, encoding='bytes')
        images.append(dict[b'data'].reshape(dict[b'data'].shape[0], 3, 32, 32))
        labels.append(dict[b'labels'])
images = np.concatenate(images)
labels = np.concatenate(labels)
images = images / 255.0
images = images.astype(np.float32)

data_images = list(zip(images, labels))
dataloader = DataLoader(data_images, batch_size=batch, shuffle=True)

# show_images(images[0][0][np.newaxis, :, :, :].transpose(0, 2, 3, 1))
# plt.show()

net = LeNet5().cuda()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters())

num = -1
for epoch in range(num_epoch):
    for d in dataloader:
        num += 1
        img = d[0]
        N = img.shape[0]
        label = d[1].numpy()
        ont_hot = np.zeros((N, num_classes))
        ont_hot[np.arange(N), label] = 1

        img = Variable(img).float().cuda()
        label = Variable(torch.from_numpy(ont_hot)).float().cuda()
        out = net(img)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if num % 50 == 0:
            out = net(Variable(torch.from_numpy(test_img)).float().cuda()).data.cpu().numpy()
            test_labels = np.argmax(out, axis=1)
            m = 0
            for lk in range(len(test_labels)):
                if test_labels[lk] == test_labs[lk]:
                    m += 1
            acc = m/len(test_labs)
#             acc = accuracy_score(test_labs, test_labels)
            print('Epoch[{}/{}], loss:{:.6f}, accuracy:{:.6f}'.format(epoch, num_epoch, loss, acc))
