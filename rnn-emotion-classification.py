import jieba
import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch import nn

# 路径
path = 'D:/cyr/datanew/datanew/neg'
# 要剔除的标点符号
punc = '.,，。;《》？！“”‘’、:\n'
# 进行预处理
for dirpath,dirnames,filenames in os.walk(path):
    for filename in filenames:
        filepath = dirpath+"/"+filename
        with open(filepath,encoding = "utf-8") as f:
            # 读取文件
            lines = f.readlines()
            seg_sentence = ''
            for line in lines:
                # 去除标点符号
                line = re.sub(r"[{}]+".format(punc)," ",line.strip())
                seg_list = jieba.cut(line)
                for word in seg_list:
                    if word != '\t':
                        seg_sentence += word + " "
            with open('output_neg.txt','a',encoding = "utf-8") as o:
                    o.write(seg_sentence + '\n')
print("finish")


# 负面的数据
data_neg = pd.read_csv('output_neg.txt',header=None)
data_neg['emotion'] = 0


# 路径
path = 'D:/cyr/datanew/datanew/pos'
# 要剔除的标点符号
punc = '.,，。;《》？！“”‘’、:\n'
# 进行预处理
for dirpath,dirnames,filenames in os.walk(path):
    for filename in filenames:
        filepath = dirpath+"/"+filename
        with open(filepath,encoding = "utf-8") as f:
            # 读取文件
            lines = f.readlines()
            seg_sentence = ''
            for line in lines:
                # 去除标点符号
                line = re.sub(r"[{}]+".format(punc)," ",line.strip())
                seg_list = jieba.cut(line)
                for word in seg_list:
                    if word != '\t':
                        seg_sentence += word + " "
            with open('output_pos.txt','a',encoding = "utf-8") as o:
                    o.write(seg_sentence + '\n')
print("finish")

data_pos = pd.read_csv('output_pos.txt',header=None)
# 正面的数据
data_pos['emotion'] = 1
data_pos
# 将正面的和负面的数据组合
data = pd.concat([data_pos,data_neg],axis=0,ignore_index=True)

# 分为训练集和测试集
x = data[0].values
y = data['emotion'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# 生成count vector
vect = CountVectorizer(max_df = 0.8,
                       min_df = 3,
                       token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b')
x_train = vect.fit_transform(x_train).toarray()

x_train = x_train.reshape(1,x_train.shape[0], x_train.shape[1])

# 构建LSTM神经网络
class lstm(nn.Module):
    def __init__(self, input_size=x_train.shape[2], hidden_size=4, output_size=1, num_layer=2):
        super(lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x


LSTM = lstm()
print(LSTM)

# 损失函数为MSE
criterion = nn.MSELoss()
# Adam优化器
optimizer = torch.optim.Adam(LSTM.parameters(), lr=1e-2)

# 训练100次
for epoch in range(100):
    var_x = torch.tensor(x_train, dtype=torch.float32)
    var_y = torch.tensor(y_train, dtype=torch.float32)
    # 前向传播
    out = LSTM(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:  # 每 2 次输出结果
        print("In epoch ", epoch + 1, " loss is", loss)

x_test = vect.transform(x_test).toarray()
x_test = x_test.reshape(1,x_test.shape[0], x_test.shape[1])
var_x_test = torch.tensor(x_test,dtype=torch.float32)
var_y_test = torch.tensor(y_test,dtype=torch.float32)
out = LSTM(var_x_test)
loss = criterion(out, var_y_test)
print("Test loss is", loss)