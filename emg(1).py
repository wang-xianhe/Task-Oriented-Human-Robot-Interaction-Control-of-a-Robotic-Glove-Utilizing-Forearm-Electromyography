import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class LSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
        super(LSTMAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        # 新增一个attention层，用于对原始输入序列进行加权求和
        self.input_attention = nn.Linear(input_size, 1)
        # 新增一个全连接层，用于对第一个attention层的output进行维度变换
        self.input_fc = nn.Linear(input_size, hidden_size)
        # 原来的LSTM层，用于对原始输入序列进行编码
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # 原来的attention层，用于对LSTM输出序列进行加权求和
        self.output_attention = nn.Linear(hidden_size, 1)
        # 新增一个全连接层，用于对两个attention层的结果进行叠加和分类
        self.fc = nn.Linear(2 * hidden_size, 2)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # h shape: (num_layers, batch_size, hidden_size)
        # c shape: (num_layers, batch_size, hidden_size)
        batch_size, seq_len, input_size = x.size()
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        # input_attention_weights shape: (batch_size, seq_len)
        input_attention_logits = self.input_attention(x)
        input_attention_weights = torch.softmax(input_attention_logits, dim=1)
        # weighted_input shape: (batch_size, input_size)
        weighted_input = torch.sum(x * input_attention_weights, dim=1)
        # weighted_input shape: (batch_size, hidden_size)
        weighted_input = self.input_fc(weighted_input)
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        lstm_output, _ = self.lstm(x, (h, c))
        # output_attention_weights shape: (batch_size, seq_len)
        output_attention_logits = self.output_attention(lstm_output)
        output_attention_weights = torch.softmax(output_attention_logits, dim=1)
        # weighted_output shape: (batch_size, hidden_size)
        weighted_output = torch.sum(lstm_output * output_attention_weights, dim=1)
        # dropout
        weighted_output = self.dropout(weighted_output)
        # concat shape: (batch_size, 2 * hidden_size)
        concat = torch.cat([weighted_input.squeeze(1), weighted_output], dim=1)
        # output shape: (batch_size, 2)
        output = self.fc(concat)
        return output
# 数据文件所在的目录
# 检查是否有CUDA支持的GPU
if torch.cuda.is_available():
  print("CUDA is available!")
  device = torch.device("cuda") # 创建一个CUDA设备对象
else:
  print("CUDA is not available.")
  device = torch.device("cpu") # 创建一个CPU设备对象


data_dir_mode1 = 'dataemg/mode1/'
data_dir_mode2='dataemg/mode2/'


# 读取所有数据文件，并将它们合并为一个三维数组
data_list1 = []
for file_name in os.listdir(data_dir_mode1):
    file_path = os.path.join(data_dir_mode1,file_name)
    df = pd.read_csv(file_path, header=None)
    df.loc[len(df)] = [0] * 8
    data = np.array(df.values)
    data_list1.append(data)

data_mode1= np.stack(data_list1)

data_list2 = []
for file_name in os.listdir(data_dir_mode2):
    file_path = os.path.join(data_dir_mode2,file_name)
    df = pd.read_csv(file_path, header=None)
    df.loc[len(df)] = [1] * 8
    data = np.array(df.values)
    data_list2.append(data)
data_mode2= np.stack(data_list2)
data=np.concatenate((data_mode1, data_mode2), axis=0)

# 分割数据和标签
X = data[:, :2000, :]
y = data[:, 2000,0]

# 将标签转换为整数类型
y = y.astype(int)

# 将数据分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# 将数据转换为张量
train_data = torch.Tensor(X_train).to(device)
train_labels = torch.LongTensor(y_train).to(device)
test_data = torch.Tensor(X_test).to(device)
test_labels = torch.LongTensor(y_test).to(device)



model = LSTMAttention(input_size=8, hidden_size=64, num_layers=2)
model=model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

num_epochs = 100
batch_size = 32
patience = 10
best_loss = float('inf')
stop_counter = 0

for epoch in range(num_epochs):
    for i in range(0, len(train_data), batch_size):
        # 获取一个batch的数据和标签
        batch_data = train_data[i:i + batch_size]
        batch_labels = train_labels[i:i + batch_size]

        # 前向传播
        outputs = model(batch_data)

        # 计算损失
        loss = criterion(outputs, batch_labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 每个epoch结束后输出损失
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    # early stopping
    if loss.item() < best_loss:
        best_loss = loss.item()
        stop_counter = 0
    else:
        stop_counter += 1

    if stop_counter >= patience:
        print('Early stopping')
        break

# 测试模型
with torch.no_grad():
    outputs = model(test_data).cpu()
    _, predicted = torch.max(outputs.data, 1)

    accuracy = accuracy_score(test_labels.cpu(), predicted.cpu())
    recall = recall_score(test_labels.cpu(), predicted.cpu())
    f1 = f1_score(test_labels.cpu(), predicted.cpu())
    auc = roc_auc_score(test_labels.cpu(), outputs.data[:, 1])

    print('Test Accuracy: {:.2f}%'.format(accuracy * 100))
    print('Test Recall: {:.2f}%'.format(recall * 100))
    print('Test F1 Score: {:.2f}'.format(f1))
    print('Test AUC: {:.2f}'.format(auc))