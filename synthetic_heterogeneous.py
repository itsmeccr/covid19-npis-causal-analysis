# -*- coding: utf-8 -*-
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#%%
class HeterogeneousChain3(Dataset):
  """
  # A -> B -> C

  # Ground Truth Model
  # 0 0 0
  # 1 0 0
  # 0 1 0

  # Let A, C be continuous and B be the categorical variable
  """
  def __init__(self, length, intervene=[], seed=None):
    self.length = length
    self.intervene = intervene
    self.seed = seed

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    np.random.seed(self.seed + index)
    A = np.random.uniform()
    B = int(A * 100) % 4 + np.random.choice([1, 0])
    C = 0.9 * B + np.random.normal(0,0.1)
    np.random.seed(None)
    return torch.tensor([A, B, C]), torch.tensor(self.intervene)
#%%
class FunctionalEstimator(nn.Module):
  def __init__(self, num_variables, num_features, target=2, ground_truth=None):
    super(FunctionalEstimator, self).__init__()
    self.num_variables = num_variables
    self.num_features = num_features
    if ground_truth is None: # fc as ground truth
      self.ground_truth = torch.ones(self.num_variables, self.num_variables).fill_diagonal_(0)
    else:
      self.ground_truth = ground_truth
    self.num_hidden = 4 * max(num_variables, num_features)
    self.num_output = 1
    self.target = target
    self.hidden1 = nn.Linear(self.num_features, self.num_hidden)
    self.output = nn.Linear(self.num_hidden, self.num_output)
 
  def forward(self, x):
    x = self.ground_truth[self.target] * x
    x = self.hidden1(x)
    x = F.relu(x)
    x = self.output(x)
    x = F.relu(x)
    return x
#%%
model = FunctionalEstimator(num_variables=3, num_features=3)
print(model)
loss_fn = torch.nn.MSELoss()
optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-3)

batch_size = 10
num_examples = 100
num_epochs = 50
chain3Dataset = HeterogeneousChain3(num_examples, seed=3)
chain3Train = DataLoader(dataset=chain3Dataset, shuffle=True, batch_size=batch_size)
chain3Test = DataLoader(dataset=chain3Dataset, shuffle=False, batch_size=len(chain3Dataset))

test_r2s = []
for epoch in range(num_epochs):
  train_loss, test_loss, test_r2= 0,0,0
  model.train()
  for (data, intGT) in chain3Train:
    optim.zero_grad()
    predict = model(data)
    predict = predict.view(predict.shape[0])
    target = data[:, model.target]
    loss = loss_fn(target, predict)
    loss.backward()
    optim.step()
    train_loss += loss

  
  with torch.no_grad():
    model.eval()
    for (data, intG) in chain3Test:
      optim.zero_grad()
      predict = model(data)
      predict = predict.view(predict.shape[0])
      target = data[:, model.target]
      loss = loss_fn(target, predict)
      test_loss += loss
      test_r2 += r2_score(target, predict)
      # if epoch == num_epochs-1:
      #   print("Target:", target)
      #   print("Predict:", predict)
    
  test_r2s.append(test_r2)
  print("Epoch {}, Train Loss: {}, Test Loss: {}, Test R2: {}".format(epoch, train_loss, test_loss, test_r2))
  
#%%
noise1 = test_r2s
noise01 = test_r2s
#%%
plt.plot(noise1, label="Noise:u(0,1)")
plt.plot(noise01, label="Noise:u(0,0.1)")
plt.xlabel("Epochs")
plt.ylabel("R2 Score on test data")
plt.title("Functional Estimator Performance for Heterogeneous Variables Causal Model")
plt.legend()
plt.show()

