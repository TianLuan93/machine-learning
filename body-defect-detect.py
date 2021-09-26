
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt

## 超参数，学习速率和训练循环次数
learning_rate = 0.0005
EPOCH=301
path = r""
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
data = []
labels = np.zeros(653)
index = 0
for dirpath,dirnames,filenames in os.walk(path):
    for filename in filenames:
        Img = cv2.imread(os.path.join(dirpath,filename))
        #resize the image of dataset to let data be smaller 90*72
        Img = cv2.resize(Img,(224,224))
        if dirpath[14:len(dirpath)] == "Liujian":
            label = 0
            labels[index] = label
            index = index + 1
        else:
            label = 1
            labels[index] = label
            index = index + 1
        Img = Img.reshape(3,224,224).astype(np.double)
        data.append(Img)
data = np.asarray(data,dtype = np.double)
# print(data.shape)
train_val_x,test_x,train_val_y,test_y = train_test_split(data,labels, train_size=0.7, test_size=0.3)
train_x, val_x, train_y, val_y = train_test_split(train_val_x, train_val_y, train_size=0.7, test_size=0.3)
# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)
# print(val_x.shape)
# print(val_y.shape)

class AlexNet(nn.Module):
    def __init__(self,num_classes=11):
        super(AlexNet,self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=2,bias=False),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
    def forward(self,x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0),256*6*6)
        x = self.classifier(x)
        return x


model = AlexNet()
train = torch.tensor(train_x)
labels = torch.tensor(train_y).long()
model.double()

model = model.to(device)
train = train.to(device)
labels = labels.to(device)
#%%
## 优化器采用adam优化器，速度比较快
#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
## 损失函数采用交叉熵函数
#loss_fun
loss_func = nn.CrossEntropyLoss()
# train_plot_x = []
# train_plot_y = []
# test_plot_x = []
# test_plot_y = []

for epoch in range(EPOCH):
    # calculate the output
    output = model(train)
    # calculate teh loss
    loss = loss_func(output, labels)
    # zero last grad
    optimizer.zero_grad()
    # back propgation
    loss.backward()
    # update the parameters
    optimizer.step()
    if epoch%10 == 0:
        prediction = torch.max(F.softmax(output.cpu()), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = labels.cpu().data.numpy()
        accuracy = sum(pred_y == target_y)/(labels.shape[0])
        # train_plot_x.append(epoch)
        # train_plot_y.append(accuracy)
        print("Train loss is",loss,"\nTrain accuracy is",100*accuracy,"%")


val = torch.tensor(val_x)
val_labels = torch.tensor(val_y).long()
output_val = model(val)
prediction_val = torch.max(F.softmax(output_val), 1)[1]
pred_y_val = prediction_val.data.numpy().squeeze()
target_y_val = val_labels.data.numpy()
accuracy = sum(pred_y_val == target_y_val) / (val_labels.shape[0])
# test_plot_x.append(epoch)
# test_plot_y.append(accuracy)
print("Validation accuracy is", 100 * accuracy, "%")

test = torch.tensor(test_x)
test_labels = torch.tensor(test_y).long()
output_test = model(test)
prediction_test = torch.max(F.softmax(output_test), 1)[1]
pred_y_test = prediction_test.data.numpy().squeeze()
target_y_test = test_labels.data.numpy()
accuracy = sum(pred_y_test == target_y_test) / (test_labels.shape[0])
# test_plot_x.append(epoch)
# test_plot_y.append(accuracy)
print("Test accuracy is", 100 * accuracy, "%")