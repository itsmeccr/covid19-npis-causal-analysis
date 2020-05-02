# -*- coding: utf-8 -*-
"""
Replicating Neural Causal Model by Bengio and Extending for 
Continuous variables
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import logging
sns.set()
#%%
class MetaNeuralCausal(nn.Module):
    """
    """
    def __init__(self, num_variables, num_features, num_hidden=None):
        
        super(MetaNeuralCausal, self).__init__()
        self.M = num_variables
        self.N = num_features
        if num_hidden is None:
            self.H = (4* self.M) if self.M > self.N else (4*self.N)
        else:
            self.H = num_hidden
        
        self.register_parameter("gamma", Parameter(torch.zeros((self.M, self.M), dtype=torch.double)))
        self.register_parameter("W0slow",  Parameter(torch.zeros((self.M, self.H, self.M, self.N), dtype=torch.double)))
        self.register_parameter("B0slow",  Parameter(torch.zeros((self.M, self.H), dtype=torch.double)))
        self.register_parameter("W1slow",  Parameter(torch.zeros((self.M, self.N, self.H), dtype=torch.double)))
        self.register_parameter("B1slow",  Parameter(torch.zeros((self.M, self.N), dtype=torch.double)))
        
        self.register_parameter("W0fast",  Parameter(torch.zeros_like(self.W0slow)))
        self.register_parameter("B0fast",  Parameter(torch.zeros_like(self.B0slow)))
        self.register_parameter("W1fast",  Parameter(torch.zeros_like(self.W1slow)))
        self.register_parameter("B1fast",  Parameter(torch.zeros_like(self.B1slow)))  
        
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.1)
        self.init_params()
        
    def init_params(self):
        for i in range(self.M):
            torch.nn.init.orthogonal_(self.W0slow[i])
            torch.nn.init.orthogonal_(self.W1slow[i])
        torch.nn.init.uniform_(self.B0slow,    -.1, +.1)
        torch.nn.init.uniform_(self.B1slow,    -.1, +.1)
        torch.nn.init.uniform_(self.gamma, -.1, +.1)
        with torch.no_grad(): self.gamma.diagonal().fill_(float("-inf"))
        
    def configpretrainiter(self, full_connect=False):
        """
        Sample a configuration for pretraining.
        """
        if not full_connect:
            yield from self.configiter()
        else:
            yield from self.configiter_pretrain()

    def configiter(self):
        """Sample a configuration from current gamma."""
        while True:
            with torch.no_grad():
                gammaexp = self.gamma.sigmoid()
                gammaexp = torch.empty_like(gammaexp).uniform_().lt_(gammaexp)
                gammaexp.diagonal().zero_()
            yield gammaexp

    def configiter_pretrain(self):
        """Sample a configuration that resembles fully connected graph."""
        while True:
            with torch.no_grad():
                gammaexp = torch.ones_like(self.gamma)
                gammaexp.diagonal().zero_()
            yield gammaexp
        
    def forward(self,sample):
        """
        

        Parameters
        ----------
        sample : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass
    
    def reconstrain_gamma(self):
        with torch.no_grad():
            self.gamma.clamp_(-5,+5)
            self.gamma.diagonal().fill_(float("-inf"))    
    
    def logits(self, sample, config):
        """
        logits of sample variables given sampled configuration.
        input  sample = (bs, M, N)  # Actual value of the sample
        input  config = (M, M)      # Configuration
        return logits = (bs, M, N)
        """
        W0 = self.W0fast+self.W0slow
        W1 = self.W1fast+self.W1slow
        B0 = self.B0fast+self.B0slow
        B1 = self.B1fast+self.B1slow
        v = torch.einsum("ihjk,ij,bjk->bih", W0, config, sample)
        v = v + B0.unsqueeze(0)
        v = self.leaky_relu(v)
        v = torch.einsum("ioh,bih->bio",     W1, v)
        v = v + B1.unsqueeze(0)
        return v
    
    
    
    def logprob(self, sample, config, block=()):
        """
        Log-probability of sample variables given sampled configuration.
        input  sample = (bs, M, N)  # Actual value of the sample
        input  config = (M, M)      # Configuration
        return logprob = (bs, M)
        """
        block = [block] if isinstance(block, int) else list(set(iter(block)))
        block = torch.as_tensor(block, dtype=torch.long, device=sample.device)
        block = torch.ones(self.M, device=sample.device).index_fill_(0, block, 0)
        v = self.logits(sample, config)
        v = v.log_softmax(dim=2)
        v = torch.einsum("bio,bio->bi", v, sample)
        vn = torch.einsum("bi,i->bi", v, 0+block)
        vi = torch.einsum("bi,i->bi", v, 1-block)
        return vn, vi
    
    def parameters_fastslow(self):
        return zip(iter([self.W0fast, self.B0fast, self.W1fast, self.B1fast]),
                   iter([self.W0slow, self.B0slow, self.W1slow, self.B1slow]))
    
    def parameters_fast(self):
        for f,s in self.parameters_fastslow(): yield f
        
    def parameters_slow(self):
        for f,s in self.parameters_fastslow(): yield s
        
    def parameters(self):
        for f,s in self.parameters_fastslow(): yield f+s
        
    def zero_fastparams(self):
        with torch.no_grad():
            for f in self.parameters_fast(): f.zero_()
        
    def structural_parameters(self):
        return iter([self.gamma])
            
#%%
            
class MetaNeuralCausalTrainer():
    def __init__(self, model, lmaxent=0.0,lsparse=0.1, ldag=0.6, cuda=False):
        self.model = model
        self.goptimizer = torch.optim.Adam(self.model.structural_parameters(),
                                     lr=5e-2, betas=(0.9, 0.999))
        self.msoptimizer = torch.optim.Adam(self.model.parameters_slow(),
                             lr=5e-2, betas=(0.9, 0.999))
        self.mfoptimizer = torch.optim.Adam(self.model.parameters_fast(), lr=5e-2, betas=(0.9, 0.999))
        self.lmaxent = lmaxent
        self.lsparse = lsparse
        self.ldag = ldag

        
        
    def train(self, dataset, batch_size, num_epochs, dataset_inv=None,
              num_inv=40, epoch_predict=1, predict_cpb=1,epoch_transfer=5,
              transfer_cpb=20, verbose=10, frac_transfer=None):
        smpiter = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)
        cfgiter = self.model.configpretrainiter(full_connect=False)
        # Pretraining "Stationary Distribution" for Wslow,Bslow
        # Allowing all other inputs to reduce bias
        epoch_pretrain = num_epochs
        for epoch in range(epoch_pretrain):
            for b, ((batch, label, intGT), config) in enumerate(zip(smpiter, cfgiter)): 
                self.msoptimizer.zero_grad()
                nll = -self.model.logprob(batch, config)[0].mean()
                nll.backward()
                self.msoptimizer.step()
                if verbose and b % verbose == 0:
                    print("Epoch:{} Train functional param only NLL: {}".format(epoch, nll.item()))
        
        # Less number of examples in adptation should be sufficient
        if frac_transfer is not None:
            dataset.length = int(frac_transfer * dataset.length)
                
        for inv in range(num_inv):
            print("Performing intervention")
            I_N = torch.randint(high=model.M, size=(1,))
            #TODO handle practical intervention
            dataset.intervene = I_N
            with torch.no_grad():
                smpiter = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)
                cfgiter = self.model.configpretrainiter(full_connect=True)
                accnll  = 0
                for epoch in range(epoch_predict):
                    for (batch, label, intGT) in smpiter:
                        for cfg, config in enumerate(cfgiter):
                            # accumulate NLL for each variable
                            accnll += -self.model.logprob(batch, config)[0].mean(0)
                            if cfg+1 == predict_cpb:
                                break
                selnode = torch.argmax(accnll).item()
                print("Predicted Intervention Node: {}  Actual Intervention Node: {}".format([selnode], I_N))
                intervention = selnode    
                
            self.goptimizer.zero_grad()
            self.model.gamma.grad = torch.zeros_like(self.model.gamma)
            
            gammagrads = [] # List of T tensors of shape (M,M,) indexed by (i,j)
            logregrets = [] # List of T tensors of shape (M,)   indexed by (i,)
            
            
            
            """Transfer Episode Adaptation Loop"""
            smpiter = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)
            for xfer_epoch in range(epoch_transfer):
                for (batch, label, intGT) in smpiter:
                    gammagrad = 0
                    logregret = 0
                    """Configurations Loop"""
                    cfgiter = self.model.configiter()
                    for xcfg, config in enumerate(cfgiter):
                        if xcfg + 1 == transfer_cpb:
                            break
                        logpn, logpi = self.model.logprob(batch, config, block=intervention)
                        with torch.no_grad():
                            gammagrad += self.model.gamma.sigmoid() - config
                            logregret += logpn.mean(0)
                        # logpi.sum(1).mean(0).backward()
                    
                    gammagrads.append(gammagrad)
                    logregrets.append(logregret)

        
            # """Update Fast Optimizer"""
            # for xfer_epoch in range(1):
            #     self.model.zero_fastparams()
            #     for b, (batch, label, intGT) in enumerate(smpiter):
            #         self.mfoptimizer.zero_grad()
            #         cfgiter = self.model.configiter()
            #         xfer_logprob = 0
            #         for xcfg, config in enumerate(cfgiter):
            #             if xcfg + 1 == transfer_cpb:
            #                 break
            #             logprob = self.model.logprob(batch, config)[0].sum(1).mean()
            #             xfer_logprob += logprob
            #             logprob.backward()
            #         print(xfer_logprob.item())
            #         self.mfoptimizer.step()
            
            
            # all_logprobs = []
            # for xfer_epoch in range(epoch_transfer):
            #     for (batch, label, intGT) in smpiter:
            #         cfgiter = self.model.configiter()
            #         for xcfg, config in enumerate(cfgiter):
            #             if xcfg + 1 == transfer_cpb:
            #                 break
            #             all_logprobs.append(self.model.logprob(batch, config)[0].mean())
            
            """Gamma Gradient Estimator"""
            with torch.no_grad():
                gammagrads = torch.stack(gammagrads)
                logregrets = torch.stack(logregrets)
                normregret = logregrets.softmax(0)
                dRdgamma   = torch.einsum("kij,ki->ij", gammagrads, normregret)
                self.model.gamma.grad.copy_(dRdgamma)
                # all_logprobs = torch.stack(all_logprobs).mean()
            
            """Gamma Regularizers"""
            siggamma = self.model.gamma.sigmoid()
            Lmaxent  = ((siggamma)*(1-siggamma)).sum().mul(-self.lmaxent)
            Lsparse  = siggamma.sum().mul(self.lsparse)
            Ldag     = siggamma.mul(siggamma.t()).cosh().tril(-1).sum() \
                               .sub(self.model.M**2 - self.model.M) \
                               .mul(self.ldag)
            
            (Lmaxent + Lsparse + Ldag).backward()
            
            """Perform Gamma Update with constraints"""
            self.goptimizer.step()
            self.model.reconstrain_gamma()
            
            with torch.no_grad():
                print(self.model.gamma)
                print(siggamma)
                # print("All log prob:  {}".format(all_logprobs.item()))
            
        
                    
    def test(self, dataset, batch_size, verbose=10):
        smpiter = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)
        cfgiter = self.model.configpretrainiter(full_connect=True)
        with torch.no_grad():
            for b, ((batch, label, intGT), config) in enumerate(zip(smpiter, cfgiter)):
                predict = F.softmax(model.logits(batch, config), dim=2)
                [print (data, cpt) for (data, cpt) in zip(batch, predict)]
                predict = predict[:,:,1].data.reshape(-1) > 0.5
                label = label.reshape(-1)           
                acc = accuracy_score(label, predict)
                print(acc)
                
        
#%%
from synthetic_categorical_dataset import CategoricalChain3, CategoricalFork3

dataset = CategoricalChain3(1000)
model = MetaNeuralCausal(num_variables=3, num_features=2)
print(model.gamma)
modelTrainer = MetaNeuralCausalTrainer(model, lmaxent=0, lsparse=0.1, ldag=0.7)
modelTrainer.train(dataset, batch_size=100, num_epochs=15, num_inv=50, frac_transfer=0.5)
#%%
dataset = CategoricalFork3(1000)
model = MetaNeuralCausal(num_variables=3, num_features=2)
print(model.gamma)
modelTrainer = MetaNeuralCausalTrainer(model, lmaxent=0, lsparse=0.1, ldag=0.7)
modelTrainer.train(dataset, batch_size=100, num_epochs=15, num_inv=50, frac_transfer=0.5)
#%%
test_dataset = CategoricalFork3(10)
modelTrainer.test(test_dataset, batch_size=len(test_dataset))


        
        