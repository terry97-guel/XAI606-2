# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import pandas as pd
import seaborn as split_rngs
from sklearn.model_selection import train_test_split



# Get Boston Housing dataset
delimiter = ','
column_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
BostonTrain = pd.read_csv("train.csv", sep=delimiter)
BostonVal = pd.read_csv("validation.csv", sep=delimiter)

# Extract values from DataFrame and convert to torch tensors
train_X = BostonTrain.drop(['MEDV'], axis=1)
train_X = torch.tensor(train_X.values, dtype=torch.float32)
train_Y = BostonTrain[['MEDV']]
train_Y = torch.tensor(train_Y.values, dtype=torch.float32)
Val_X = BostonVal.drop(['MEDV'], axis=1)
Val_X = torch.tensor(Val_X.values, dtype=torch.float32)
Val_Y = BostonVal[['MEDV']]
Val_Y = torch.tensor(Val_Y.values, dtype=torch.float32)
print(Val_X.shape)




# Define a simple linear regression model2
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(13, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.linear(x)

# Create an ensemble of models
num_ensembles = 3
ensemble = [LinearRegressionModel() for _ in range(num_ensembles)]

# Define loss function and optimizer for each model
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizers = [torch.optim.Adam(model.parameters(), lr=0.01) for model in ensemble]

# Training loop for each model
num_epochs = 2000
val_acc_ls = []
for epoch in range(num_epochs):
    for model, optimizer in zip(ensemble, optimizers):
        # Forward pass
        outputs = model(train_X)
        # print(outputs)
        loss = criterion(outputs, train_Y)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        # calculate the accuracy(MSE error) of model for validation data
        predictions = torch.stack([model(Val_X) for model in ensemble])
        
        # choose the prediction value: mean or median (in here, I used mean)
        pred_val = torch.mean(predictions, axis=0)
        val_acc = criterion(pred_val, Val_Y)
        val_acc_ls.append(val_acc.item())
        print(f"Epoch [{epoch}/{num_epochs}], Acc(MSE): {val_acc.item()}")

# =================================================================
# make a code for testing
# 1. predict the values for given test set (shape will be [num_ensemble, batch_size, 1(value)])
# 2-1. choose the prediction value for free (it can be mean or median for each model -> shape will be [batch_size, 1])
# 2-2. calculate the uncertainty(variance) for each model (shape will be [batch_size, 1])
# 3. submit the results of 2-1 and 2-2 for 1 csv file
# there will be 2 columns. first column will be preiction value and second column will be uncertainty(variance)
# ==================================================================
# %%
from matplotlib import pyplot as plt
plt.plot(val_acc_ls)
# %%
