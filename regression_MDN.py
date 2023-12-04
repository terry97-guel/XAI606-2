# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import torch.optim as optim
import pandas as pd
from typing import Tuple, Union
from torch.distributions import OneHotCategorical, Normal, MultivariateNormal, Laplace

# set random seed
random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True


# Get Boston Housing dataset
delimiter = ','
column_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
BostonTrain = pd.read_csv("train.csv", sep=delimiter)
BostonVal = pd.read_csv("validation.csv", sep=delimiter)

# Extract input and output data and convert to torch tensors
train_X = BostonTrain.drop(['MEDV'], axis=1)
train_X = torch.tensor(train_X.values, dtype=torch.float32)
train_Y = BostonTrain[['MEDV']]
train_Y = torch.tensor(train_Y.values, dtype=torch.float32)
Val_X = BostonVal.drop(['MEDV'], axis=1)
Val_X = torch.tensor(Val_X.values, dtype=torch.float32)
Val_Y = BostonVal[['MEDV']]
Val_Y = torch.tensor(Val_Y.values, dtype=torch.float32)


input_std, input_mean = torch.std_mean(train_X, dim=0)
output_std, output_mean = torch.std_mean(train_Y, dim=0)

normalized_train_X = (train_X - input_mean) / input_std
normalized_train_Y = (train_Y - output_mean) / output_std
normalized_val_X = (Val_X - input_mean) / input_std
normalized_val_Y = (Val_Y - output_mean) / output_std


# sample input data for MIMO Regression model: independent for each heads
def sample_data(X, Y, batch_size):
    indices = torch.randperm(len(X))[:batch_size]
    sampled_X_batch = X[indices]
    sampled_Y_batch = Y[indices]

    return sampled_X_batch, sampled_Y_batch

X_tensor, Y_tensor = sample_data(train_X, train_Y, batch_size=48)


def get_linear_layer(hdim, hidden_actv, last_actv = None, std=0.1):    
    layers = []
    for hdim_idx in range(0,len(hdim)-1):
        layer = nn.Linear(hdim[hdim_idx],hdim[hdim_idx+1])
        # torch.nn.init.normal_(layer.weight,.0,std)
        # torch.nn.init.xavier_normal_(layer.weight)
        layers.append(layer)
        
        if hdim_idx == len(hdim)-2:
            if last_actv is not None:
                layers.append(last_actv)
        else:
            layers.append(hidden_actv)
    return layers

class MDN(nn.Module):
    """
    Mixture density network.

    [ Bishop, 1994 ]

    n_components: int; number of components in the mixture model
    """
    def __init__(self,x_dim,n_components = 3):
        super().__init__()
        self.pi_network = CategoricalNetwork(
            x_dim, n_components = n_components)
        
        self.normal_network = MixtureComponentNetwork(
            x_dim, n_components = n_components)

    def forward(self, x) -> Tuple[OneHotCategorical, Union[Normal,Laplace]]:
        return self.pi_network(x), self.normal_network(x)

    def sample(self, x):
        '''
        When sampling, we use mode of MDN components.
        Instead of sampling from MDN components 
        '''
        pi, normal = self.forward(x)
        samples = torch.sum(pi.sample().unsqueeze(2) * normal.loc, dim=1)

        return samples

class MixtureComponentNetwork(nn.Module):
    def __init__(self, x_dim, hdim=[64,128,64],n_components = 3):
        super().__init__()
        self.n_components = n_components
        hdim = list(hdim)
        hdim.insert(0,x_dim)        
        self.layers          = nn.Sequential(*get_linear_layer(list(hdim), hidden_actv=nn.ReLU(), last_actv=nn.ReLU()))
        
        self.y_loc       = nn.Sequential(*get_linear_layer(hdim=[hdim[-1], n_components], hidden_actv=nn.ReLU()))
        self.y_scale     = nn.Sequential(*get_linear_layer(hdim=[hdim[-1], n_components], hidden_actv=nn.ReLU(), last_actv=nn.Sigmoid()))
        

    def forward(self, x) -> Union[Normal, Laplace]:
        '''
        Latent variable z and conditional vector c are fed to decoder,
        resulting anchor distribution of anchor_loc(mean), anchor_scale(std).

        anchor_loc is scaled according to joint limit.
        anchor_scale is scaled in similar to SIGMA VAE(https://arxiv.org/pdf/2006.13202.pdf)

        To leverge auto-calibration of Beta-VAE, SIGMA VAE is used for every MDN components. 
        Possible betas range (0.5~2)
        
        '''
        # Feed
        out = self.layers(x)

        y_loc   = self.y_loc(out).reshape(-1, self.n_components, 1)
        y_scale = self.y_scale(out).reshape(-1, self.n_components, 1)
        
        betas = [0.5, 2]
        lower_bound = torch.sqrt(torch.tensor(betas[0])); upper_bound = torch.sqrt(torch.tensor(betas[1]))
        y_scale = y_scale * (upper_bound-lower_bound)/2 + (upper_bound+lower_bound)/2 
        
        # y_scale = torch.ones_like(y_scale)
        assert (y_scale >= 0).all()
        
        # Get anchor distribution
        y_dist = Normal(y_loc,y_scale)

        return y_dist


def loss_mdn(Mixture_Mode, Mixture_components, y):
    loglik  = Mixture_components.log_prob(y.unsqueeze(1).expand_as(Mixture_components.loc))
    loglik  = torch.sum(loglik, dim=2)
    weighted_NLL = -torch.logsumexp(Mixture_Mode.logits + loglik, dim=1)
    
    return weighted_NLL.mean()

class CategoricalNetwork(nn.Module):
    def __init__(self,xdim, hdim=[64,128,64], n_components = 5):
        super().__init__()
        self.n_components = n_components
        self.xdim   = xdim
        hdim = list(hdim)
        hdim.insert(0,xdim)
        
        layers            = get_linear_layer(list((hdim)), hidden_actv=nn.ReLU())
        self.layers       = nn.Sequential(*layers)
        
        self.MixtureLogit = nn.Sequential(*get_linear_layer(hdim=[hdim[-1], n_components], hidden_actv=nn.ReLU()), nn.Softmax(dim=1))
        
    def forward(self, x) -> OneHotCategorical:
        out = self.layers(x)
        probs = self.MixtureLogit(out)
        
        assert torch.sum(probs, dim=1).allclose(torch.ones(probs.shape[0]))
        return OneHotCategorical(probs=probs)


        
        
x_dim = train_X.shape[1]
model = MDN(x_dim=x_dim,n_components=5)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 2_000 * 3

val_acc_ls = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    normalized_X_tensor, normalized_Y_tensor = sample_data(normalized_train_X, normalized_train_Y, batch_size=48)
    
    # Forward pass
    Mixture_Mode, Mixture_components = model(normalized_X_tensor)
    # Compute loss
    loss = loss_mdn(Mixture_Mode, Mixture_components, normalized_Y_tensor)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        pred_val = model.sample(normalized_val_X)
        pred_val = pred_val * output_std + output_mean
        val_acc = criterion(pred_val, Val_Y)


        # pred_val = model.sample(normalized_train_X)
        # pred_val = pred_val * output_std + output_mean
        # val_acc = criterion(pred_val, train_Y)


        print(f"Epoch [{epoch}/{num_epochs}], Acc(MSE): {val_acc.item()}")
        
        if epoch %3 == 0:
            val_acc_ls.append(val_acc.item())
    
# =================================================================
# make a code for testing
# 1. predict the values for given test set (shape will be [batch_size, num_heads, 1(value)])
# 2-1. choose the prediction value for free (it can be mean or median for each heads -> shape will be [batch_size, 1])
# 2-2. calculate the uncertainty(variance) for each heads (shape will be [batch_size, 1])
# 3. submit the results of 2-1 and 2-2 for 1 csv file
# there will be 2 columns. first column will be preiction value and second column will be uncertainty(variance)
# ==================================================================
# %%
from matplotlib import pyplot as plt
plt.plot(val_acc_ls)
# %%
