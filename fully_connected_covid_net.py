import sklearn
import numpy as np
np.set_printoptions(suppress=True)
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler, BatchSampler
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# csv file should be pre-sorted in ascending order on the ApproxDays feature
path = 'trimmed_covid_data.csv'


file = open(path, newline='')
reader = csv.reader(file)
header = next(reader)
data = [line for line in reader]
data_list = []
for sample in data:
    tmp_sample = [float(feature) for feature in sample]
    data_list.append(tmp_sample)
data_list = np.asarray(data_list, dtype='float')
COVID_data = data_list[:, :-1]
COVID_label = data_list[:, -1]

#shuffle_idx = np.random.permutation(len(COVID_data))

train_split = int(0.8*len(COVID_data))

# train split
COVID_data_train = COVID_data[:train_split]
COVID_label_train = COVID_label[:train_split]

# test_split
COVID_data_test = COVID_data[train_split:]
COVID_label_test = COVID_label[train_split:]

print(COVID_data.shape, COVID_label.shape)

#reg = LinearRegression().fit(COVID_data_train, COVID_label_train)
#print('linear regression coeffs', np.array(reg.coef_, dtype='float'))


class COVID_dataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.tensor(data).double()
        self.label = torch.tensor(label).double()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class COVID_fC(nn.Module):
    def __init__(self):
        super(COVID_fC, self).__init__()
        # initialize joint layers
        self.fc1 = nn.Linear(17, 128).double()
        self.fc2 = nn.Linear(128, 17).double()
        #self.fc2 = nn.Linear(128, 17).double()
        self.fc3 = nn.Linear(17, 1).double()

    def forward(self, X):
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        X = F.relu(X)
        X = self.fc3(X)
        return X


model = COVID_fC()
model_name = 'covid_fc_model'
epoch = 500
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
train_dataset = COVID_dataset(COVID_data_train, COVID_label_train)
# sequential sampler to treat data like time-series
sampler = BatchSampler(SequentialSampler(train_dataset), batch_size=len(train_dataset), drop_last=False)
train_loader = DataLoader(dataset=train_dataset, batch_sampler=sampler)

for i in range(epoch):
    for (data,label) in train_loader:

        optimizer.zero_grad()
        pred = model(data)
        pred = pred.view(pred.size()[0])
        Loss = loss_fn(pred, label)

        Loss.backward()
        optimizer.step()
    print(Loss.item())
    print(model.fc3.weight.data.numpy())
    torch.save(model.state_dict(), model_name)


test_dataset = COVID_dataset(COVID_data_test, COVID_label_test)
# sequential sampler to treat data like time-series
sampler = BatchSampler(SequentialSampler(test_dataset), batch_size=len(test_dataset), drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_sampler=sampler)

for (data,label) in test_loader:
    with torch.no_grad():
        preds = model(data)
        preds = preds.view(preds.size()[0])

        mse = loss_fn(preds, label)
        print('mse value : ', mse)
        r2 = r2_score(label.data.numpy(), preds.data.numpy())
        print('r2 score : ', r2)