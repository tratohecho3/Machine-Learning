import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from numpy.core.umath_tests import inner1d

import hdnntools as hdn
import pyanitools as at

def transform_input(X):
    d = Variable(torch.FloatTensor(X.size()[0],3))
    for m in range(X.size()[0]):
        it = 0
        for i in range(X.size()[1]):
            for j in range(i+1,X.size()[1]):
                d[m,it] = torch.sqrt(torch.sum((X[m,i]-X[m,j])**2))
                it+=1

    return d

# network class
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(3,  2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2,  1)

    def forward(self, X):
        X = transform_input(X)

        X = F.elu(self.fc1(X))
        X = F.elu(self.fc2(X))
        X = self.fc3(X)
        return X

# Get data set
dataset = '/home/jujuman/Research/GDB-11-test-LOT/ani-gdb-c01.h5'
data = at.anidataloader(dataset).get_data('/gdb11_s01/mol00000020')

x = torch.from_numpy(data['coordinates'])
y = torch.from_numpy(np.array(data['energies']-np.array(data['energies'].min()), dtype=np.float32))

#x = x.cuda()
#y = y.cuda()

print(x.numpy())
print(y.numpy())

# Define network
net = Net()
#net = net.cuda()
# create your optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for i in range(10000):
    input = Variable(x) # convert input to Var
    target = Variable(y)  # a dummy target, for example

    net.zero_grad()     # zeroes the gradient buffers of all parameters

    output = net(input)  # Calculate output

    loss = criterion(output, target)
    loss.backward()     # Calc grads
    optimizer.step()    # Does the update

    print(hdn.hatokcal*np.mean(np.abs(output.data.numpy().flatten()-target.data.numpy().flatten())))