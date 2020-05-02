# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.utils.data import Dataset

class CategoricalChain3(Dataset):
  """
  # A -> B -> C

  # Ground Truth Model
  # 0 0 0
  # 1 0 0
  # 0 1 0

  # P(A=1) = 0.7
  # P(B=1|A=1) = 0.2, P(B=1|A=0) = 0.6
  # P(C=1|B=1) = 0.8, P(C=1|B=0) = 0.4
  
  # Intervention Change A's or B's value
  """
  def __init__(self, length, intervene=[], noise=1, seed=1):
    self.length = length
    if type(intervene) == list:
        self.intervene = torch.tensor(intervene)
    else:
        self.intervene = intervene
    
    self.seed = seed
    self.noise = noise
    
    self.functions = [
        lambda x: np.random.binomial(1,p=0.7), # A~U(0,1)
        lambda A: np.random.binomial(1,p=0.2) if A==1 else np.random.binomial(1,p=0.6), # B = f(A, N) (Categorical)
        lambda B: np.random.binomial(1,p=0.8) if B==1 else np.random.binomial(1,p=0.4)
        ]
    self.interventions = [
        lambda x: np.random.binomial(1,p=0.2),
        lambda x: np.random.binomial(1,p=0.5), 
        lambda x: np.random.binomial(1,p=0.8) if x==1 else np.random.binomial(1,p=0.4)
        ]
    self.labels = torch.eye(2) # one-hot

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    np.random.seed(self.seed + index)
    nodes = torch.empty(3,2, dtype=torch.double)
    labels = torch.empty(3, dtype=torch.long)
    previous = 0
    for i in range(3):
        if len(self.intervene) > 0 and self.intervene[0] == i:
            value = self.interventions[i](previous)
        else:
            value = self.functions[i](previous) 
        previous = value
        nodes[i]=self.labels[value]
        labels[i] = value
    np.random.seed(None)
    return nodes, labels, self.intervene

class CategoricalFork3(Dataset):
  """
  # B <- A -> C
  # A, B, C index 0,1,2
  # Ground Truth Model
  # 0 0 0
  # 1 0 0
  # 1 0 0

  # P(A=1) = 0.7
  # P(B=1|A=1) = 0.2, P(B=1|A=0) = 0.6
  # P(C=1|A=1) = 0.8, P(C=1|A=0) = 0.4
  
  # Intervention Change A's or B's value
  """
  def __init__(self, length, intervene=[], noise=1, seed=1):
    self.length = length
    if type(intervene) == list:
        self.intervene = torch.tensor(intervene)
    else:
        self.intervene = intervene
    
    self.seed = seed
    self.noise = noise
    
    self.functions = [
        lambda x: np.random.binomial(1,p=0.7), # A~U(0,1)
        lambda A: np.random.binomial(1,p=0.2) if A==1 else np.random.binomial(1,p=0.6), # B = f(A, N) (Categorical)
        lambda A: np.random.binomial(1,p=0.8) if A==1 else np.random.binomial(1,p=0.4)
        ]
    self.interventions = [
        lambda x: np.random.binomial(1,p=0.2),
        lambda x: np.random.binomial(1,p=0.5), 
        lambda x: np.random.binomial(1,p=0.4) 
        ]
    self.labels = torch.eye(2) # one-hot

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    np.random.seed(self.seed + index)
    nodes = torch.empty(3,2, dtype=torch.double)
    labels = torch.empty(3, dtype=torch.long)
    previous = 0
    for i in range(3):
        if len(self.intervene) > 0 and self.intervene[0] == i:
            value = self.interventions[i](previous)
        else:
            value = self.functions[i](previous) 
        nodes[i]=self.labels[value]
        labels[i] = value
        previous = labels[0]
    np.random.seed(None)
    return nodes, labels, self.intervene

