import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import os

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)
    return (data - min)/(max-min)

path = ""


train_data = []
train_labels = np.zeros(9866)
index = 0
#加载训练数据集，resize为28*28
for dirpath,dirnames,filenames in os.walk(path+"/training"):
    for filename in filenames:
        Img = cv2.imread(os.path.join(dirpath,filename),0)
        Img = cv2.resize(Img,(28,28))
        i = filename.index("_")
        label = filename[0:i]
        train_labels[index] = label
        index = index + 1
        Img = Img.reshape(1,28,28).astype(np.double)
        train_data.append(Img)

#%%
#
# test_data = []
# test_labels = np.zeros(3347)
# index = 0
# #加载测试数据集，resize为28*28
# for dirpath,dirnames,filenames in os.walk(path+"/testing"):
#     for filename in filenames:
#         Img = cv2.imread(os.path.join(dirpath,filename))
#         #resize the image of dataset to let data be smaller 90*72
#         Img = cv2.resize(Img,(28,28))
#         i = filename.index("_")
#         label = filename[0:i]
#         test_labels[index] = label
#         index = index + 1
#         Img = Img.reshape(3,28,28).astype(np.double)
#         test_data.append(Img)

train_data = np.asarray(train_data,dtype = np.double)
# test_data = np.asarray(test_data,dtype = np.double)

# 归一化
train_data = minmaxscaler(train_data)
# test_data = minmaxscaler(test_data)

x_train = torch.tensor(train_data, dtype=torch.float32).to(device)
y_train = torch.tensor(train_labels, dtype=torch.float32).long().to(device)

# #%%
# validation_data = []
# validation_labels = np.zeros(3430)
# index = 0
# #加载验证数据集，resize为28*28
# for dirpath,dirnames,filenames in os.walk(path+"/validation"):
#     for filename in filenames:
#         Img = cv2.imread(os.path.join(dirpath,filename),0)
#         #resize the image of dataset to let data be smaller 90*72
#         Img = cv2.resize(Img,(28,28))
#         i = filename.index("_")
#         label = filename[0:i]
#         validation_labels[index] = label
#         index = index + 1
#         Img = Img.reshape(1,28,28).astype(np.double)
#         validation_data.append(Img)
#
# validation_data = np.asarray(validation_data,dtype = np.double)
# validation_data = minmaxscaler(validation_data)
# x_val = torch.tensor(validation_data, dtype=torch.float32).to(device)
# y_val = torch.tensor(validation_labels, dtype=torch.float32).long().to(device)
#%%

iteration = 100
lr = 0.01

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(3)
        self.out = nn.Linear(32 * 4 * 4, 11)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.shape[0], 32 * 4 * 4)
        self.feature = x
        output = self.out(x)
        return output

dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
print("data loaded!")

model = CNN().to(device)
print("Model has been built!")
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
print(model)
#%%
train_loss_plot_x = []
train_loss_plot_y = []
train_accuracy_plot_x = []
train_accuracy_plot_y = []
for epoch in range(iteration):
    for batch_x, batch_y in train_loader:
        output = model(batch_x)
        optimizer.zero_grad()
        loss = loss_func(output, batch_y)
        loss.backward()
        optimizer.step()
    output = model(x_train[0:200])
    loss = loss_func(output, y_train[0:200])
    prediction = torch.max(F.softmax(output), 1)[1].cpu()
    pred_y = prediction.data.numpy().squeeze()
    target_y = y_train[0:200].cpu().data.numpy()
    accuracy = sum(pred_y == target_y) / (x_train[0:200].shape[0])
    print("In epoch", epoch + 1)
    print("Train loss is ",loss)
    print("Train accuracy is ", accuracy)
    train_loss_plot_x.append(epoch)
    train_loss_plot_y.append(loss)
    train_accuracy_plot_x.append(epoch)
    train_accuracy_plot_y.append(accuracy)



plt.figure()
plt.title("train loss")
plt.plot(train_loss_plot_x, train_loss_plot_y)
plt.show()
plt.figure()
plt.title("train accuracy")
plt.plot(train_accuracy_plot_x, train_accuracy_plot_y)
plt.show()