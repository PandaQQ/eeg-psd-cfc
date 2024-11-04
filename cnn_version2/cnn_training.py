# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:55:11 2024

@author: Qi Zheng
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
import os
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
device = torch.device("mps")

batch_size = 128

# temp = loadmat('./EEG_mental_states.mat')
temp = loadmat('./EEG_mental_states_python.mat')

data = temp['cwt_data']
label = temp['labels']
del temp

data = torch.tensor(data, device=device, dtype=torch.float)
data = torch.permute(data, (3, 0, 1, 2))
label = torch.tensor(label, device=device, dtype=torch.long)

# temp = torch.randperm(900)
# data = data[torch.cat((temp[:480], torch.arange(900, 1380))), :, :, :]
# label = label[torch.cat((temp[:480], torch.arange(900, 1380)))]
#
# label = torch.tensor(label, device=device, dtype=torch.long).squeeze()
#
# temp = torch.randperm(960)
# data_train = data[temp[:600], :, :, :]
# data_test = data[temp[600:], :, :, :]
# label_train = label[temp[:600]]
# label_test = label[temp[600:]]


data_set_size = 1300

temp = torch.randperm(1800)  # Adjusted for twice the amount of data
data = data[torch.cat((temp[:1300], torch.arange(1800, 3100))), :, :, :]  # Adjusted indices
label = label[torch.cat((temp[:1300], torch.arange(1800, 3100)))]

label = torch.tensor(label, device=device, dtype=torch.long).squeeze()

temp = torch.randperm(2600)  # Adjusted for twice the amount of data
data_train = data[temp[:data_set_size], :, :, :]  # Adjusted sizes
data_test = data[temp[data_set_size:], :, :, :]
label_train = label[temp[:data_set_size]]
label_test = label[temp[data_set_size:]]




# conv1 = nn.Conv2d(10, 10, (51,1))
# pool1 = nn.MaxPool2d((1,10))
# conv2 = nn.Conv2d(10, 1, (1,1))
# pool2 = nn.MaxPool2d((1,2))
# conv3 = nn.Conv2d(50, 1, (5,10))
# pool3 = nn.MaxPool2d((1,1))


# temp1 = conv1(data_train)
# temp2 = pool1(temp1)
# temp3 = conv2(temp2)
# temp4 = pool2(temp3)
# temp5 = conv3(temp4)
# temp6 = pool3(temp5)


class my_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(10, 20, (44, 4))
        self.bn1 = nn.BatchNorm2d(20, affine=False)
        self.bn2 = nn.BatchNorm2d(10, affine=False)
        self.bn3 = nn.BatchNorm2d(1, affine=False)
        self.pool1 = nn.MaxPool2d((1, 10))
        self.conv2 = nn.Conv2d(20, 10, (1, 1))
        self.conv3 = nn.Conv2d(5, 1, (5, 10))
        self.pool2 = nn.MaxPool2d((1, 2))
        self.pool3 = nn.MaxPool2d((1, 1))
        # self.fc1 = nn.Linear(10*12*1,32)
        self.fc1 = nn.Linear(70, 32)
        self.fc2 = nn.Linear(32, 2)
        self.drop = nn.Dropout(p=0.0)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        # x = self.drop(self.fc1(x))
        x = self.fc2(x)
        return x
    # def __init__(self):
    #     super().__init__()
    #     self.conv1 = nn.Conv2d(10, 20, (11, 1))
    #     self.bn1 = nn.BatchNorm2d(20, affine=False)
    #     self.bn2 = nn.BatchNorm2d(10, affine=False)
    #     self.pool1 = nn.MaxPool2d((1, 10))
    #     self.conv2 = nn.Conv2d(20, 10, (1, 1))
    #     self.pool2 = nn.MaxPool2d((1, 2))
    #
    #     # Placeholder layer to find the flattened size
    #     self._to_linear = None
    #     self.find_flattened_size()
    #
    #     self.fc1 = nn.Linear(self._to_linear, 32)
    #     self.fc2 = nn.Linear(32, 2)
    #     self.drop = nn.Dropout(p=0.1)
    #
    # def find_flattened_size(self):
    #     # Create a dummy input tensor with the same dimensions as your data
    #     x = torch.rand(1, 10, 44, 150)  # Adjust the size based on your input
    #     x = self.pool1(F.relu(self.bn1(self.conv1(x))))
    #     x = self.pool2(F.relu(self.bn2(self.conv2(x))))
    #     self._to_linear = x.numel()  # Total number of elements


def get_data(batch_size, g):
    idx = torch.randint(0, data_set_size - batch_size, (1,))
    x = torch.empty((batch_size, 10, data_train.shape[2], data_train.shape[3]), device=device)
    y = label_train[idx:idx + batch_size]
    for j in range(batch_size):
        temp_label = y[j]
        temp_index = (label_train == temp_label).nonzero()
        idx = torch.randint(0, temp_index.shape[0], (g,))
        x_temp = data_train[temp_index[idx, 0], :, :, :]
        x[j, :, :, :] = torch.mean(x_temp, 0, keepdim=True)
    return x, y


# def get_loss():
#     out = {}
#     model.eval()
#     eval_iter = 300
#     for split in ['train','test']:
#         for j in range(eval_iter):
#             if split == 'train':
#                 idx = torch.randint(0,600-eval_iter,(1,))
#                 x = data_train[idx:idx+eval_iter,:,:,:]
#                 y = label_train[idx:idx+eval_iter]
#                 logits = model(x)
#                 logits1 = F.softmax(logits,1)
#                 matched = torch.argmax(logits,dim=1)
#                 out['train'] = sum(matched==y[:,0])*100/eval_iter
#             if split == 'test':
#                 idx = torch.randint(0,360-eval_iter,(1,))
#                 x = data_test[idx:idx+eval_iter,:,:,:]
#                 y = label_test[idx:idx+eval_iter]
#                 logits = model(x)
#                 logits1 = F.softmax(logits,1)
#                 matched = torch.argmax(logits,dim=1)
#                 out['test'] = sum(matched==y[:,0])*100/eval_iter
#     model.train()
#     return out


def get_loss():
    out = {}
    model.eval()
    with torch.no_grad():
        for split in ['train', 'test']:
            if split == 'train':
                x = data_train
                y = label_train
            else:
                x = data_test
                y = label_test
            logits = model(x)
            predicted = torch.argmax(logits, dim=1)
            correct = (predicted == y).sum().item()
            total = y.size(0)
            accuracy = correct * 100.0 / total
            out[split] = accuracy
    model.train()
    return out


model = my_cnn()
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)


print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

loss_track = []
accu_track_train = []
accu_track_test = []

for j in range(2000):
    # idx = torch.randint(0,7000-batch_size,(batch_size,))
    # x = data_train[idx,:,:,:]
    # y = label_train[idx]

    # if j < 100:
    #     x,y = get_data(batch_size,60)
    # elif j < 500:
    #     x,y = get_data(batch_size,30)
    # elif j < 2000:
    #     x,y = get_data(batch_size,15)
    # elif j < 5000:
    #     x,y = get_data(batch_size,5)
    # else:
    #     x,y = get_data(batch_size,1)
    x, y = get_data(batch_size, 1)
    logits = model(x)
    # logits1 = F.softmax(logits,1)
    # loss = F.cross_entropy(logits1,y[:,0])
    loss = F.cross_entropy(logits, y)
    if j % 5 == 0:
        print(j)
        print('loss: ')
        print(loss.item())
        loss_track.append(loss.item())
    if j % 20 == 0:
        L = get_loss()
        accu_track_train.append(L['train'])
        accu_track_test.append(L['test'])
        print('train accuracy: ')
        print(L['test'])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # scheduler.step()

# Save the trained model

plt.subplot(311)
plt.plot(loss_track)
plt.subplot(312)
plt.plot(accu_track_train)
plt.subplot(313)
plt.plot(accu_track_test)
plt.show()

# save the model

torch.save(model.state_dict(), 'my_cnn_model_2.pth')
print("Model saved to my_cnn_model.pth")
