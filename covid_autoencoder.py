import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import csv
np.set_printoptions(threshold=sys.maxsize, suppress=True)
random.seed(123)


class COVID_dataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data).double()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.fc1 = nn.Linear(11, 6).double()
        self.fc2 = nn.Linear(6, 1).double()

        self.fc3 = nn.Linear(1, 6).double()
        self.fc4 = nn.Linear(6, 11).double()

    def forward(self, x):
        # encoder
        x = self.fc1(x)
        x = F.relu(x)
        code = self.fc2(x)

        # decoder
        x = self.fc3(code)
        x = F.relu(x)
        x = self.fc4(x)
        return x, code


if __name__ == "__main__":

    path = 'trimmed_covid_data.csv'
    file = open(path, newline='')
    reader = csv.reader(file)
    header = next(reader)
    print(header)
    data = [line for line in reader]
    data_list = []
    for sample in data:
        sample = np.asarray(sample)
        sample = sample[4:15]  # 11 features to calculate latent transmission variable
        tmp_sample = [float(feature) for feature in sample]
        data_list.append(tmp_sample)
    data_list = np.asarray(data_list, dtype='float')
    print(np.shape(data_list))

    train_split = int(0.8*len(data_list))

    # train split
    data_train = data_list[:train_split]
    print(np.shape(data_train))

    # test_split
    data_test = data_list[train_split:]
    print(np.shape(data_test))

    model = encoder()
    model_name = 'transmission_model'
    epoch = 10
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    train_dataset = COVID_dataset(data_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=200)

    # training
    for i in range(epoch):
        for data in train_loader:
            optimizer.zero_grad()
            reconstruct, code = model(data)
            Loss = loss_fn(reconstruct, data)

            Loss.backward()
            optimizer.step()
        print(Loss.item())
        print(model.fc2.weight.data.numpy())
        torch.save(model.state_dict(), model_name)

    test_dataset = COVID_dataset(data_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))

    with torch.no_grad():
        for data in test_loader:
            rec, code = model(data)
            print(np.asarray(code))
            mse = loss_fn(rec, data)
            print('mse value : ', mse)
