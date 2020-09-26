import torch
import torch.nn.functional as F
import numpy as np
import os

# Create a 2x2 matrix.
a = torch.FloatTensor([[1, 2], [3, 4]])
print(f'a: {a}\n')
print(f'first row: {a[0]}\n')
print(f'first column: {a[:, 0]}\n')

# Create a 2x1 matrix.
a = torch.FloatTensor([-1, 1]).view(2, 1)
print(f'a has a shape of {a.shape}, this means that it has {a.shape[0]} rows and {a.shape[1]} columns.')
print(f'a:\n {a}')
# Create a 1x2 matrix.
b = torch.FloatTensor([0, 1]).view(1, 2)
print(f'b has a shape of {b.shape}, this means that it has {b.shape[0]} rows and {b.shape[1]} columns.')
print(f'b:\n {b}')
# Perform matrix multiplication on a and b.
# The result of an AxB matrix multiplied with a BxC matrix is a AxC matrix.
c = a @ b
print(f'c has a shape of {c.shape}, this means that it has {c.shape[0]} rows and {c.shape[1]} columns.')
print(f'c:\n {c}')
print('----------------------------------------------------------------------------------------------')
a = torch.arange(24).view(4, 2, 3)  # Create a tensor of shape 4x2x3.
# torch.arange creates a 1d tensor of integers ranging from 0 up to the specified number.
print(f'a:\n {a}')

a_sum = a.sum(dim=1)  # The sum function adds up all the values along the specified dimension.
# In this case we sum over the second dimension.
print(f'a_sum has a shape of {a_sum.shape}')
print(f'a_sum: {a_sum}')
print('----------------------------------------------------------------------------------------------')
# Create a 2x1 matrix.
a = torch.FloatTensor([2, 4]).view(1, 2)
print(f'ab: {a}')
# Create a 1x2 matrix.
b = torch.FloatTensor([1, -1]).view(2, 1)
print(f'ab: {b}')
# This symbol * means element-wise multiplication (not to be confused with matrix multiplication!).
ab = a * b
print(f'ab: {ab}')
print('----------------------------------------------------------------------------------------------')

a = torch.FloatTensor([1, 3])
b = torch.FloatTensor([[2, 1], [-1, 4]])

# Compute your ab here. It should give the same result as a@b
ab = (a.reshape(2, 1) * b).sum(dim=0)

print(ab)
print(a @ b)
print('----------------------------------------------------------------------------------------------')

a = torch.FloatTensor([-1, 2]).view(2, 1)
b = torch.FloatTensor([2, 3]).view(2, 1)
c = torch.FloatTensor([4, 1]).view(2, 1)

concatenation0 = torch.cat([a, b, c], dim=0)
print(f'concatenate dimension 0: {concatenation0}\n')

concatenation1 = torch.cat([a, b, c], dim=1)
print(f'concatenate dimension 1: {concatenation1}')
print('----------------------------------------------------------------------------------------------')

# One very common tensor operation is concatenation.
# Concatenation means to join multiple tensors together end-to-end to create one tensor.
a = torch.FloatTensor([-1, 2]).view(2, 1)
b = torch.FloatTensor([2, 3]).view(2, 1)
c = torch.FloatTensor([4, 1]).view(2, 1)

concatenation0 = torch.cat([a, b, c], dim=0)
print(f'concatenate dimension 0: {concatenation0}\n')

concatenation1 = torch.cat([a, b, c], dim=1)
print(f'concatenate dimension 1: {concatenation1}')


class LinearRegressor(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W = torch.nn.Linear(d_in, d_out)
        # This creates a "linear" module. The linear module consists of a weight matrix [d_in, d_out] and a bias
        # vector [d_out]. When a new instance of this class is created this module will be initialized with random
        # values. Since this class subclasses torch.nn.Module, all of the parameters in the linear module will be
        # added to this LinearRegressor's parameters.

    def forward(self, x):
        # The forward function is applied whenever this model is called on some input data.
        # The forward function specifies how the model computes its output.
        y_h = self.W(x)  # Apply our linear operation to the input.
        return y_h


model = LinearRegressor(dim, 1)

# Create a stochastic gradient descent optimizer, and give it all our model's parameters as tensors to be optimized
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def train_model(model):
    n_epochs = 5
    # Train our model for n_epoch epochs
    for i in range(n_epochs):
        yh = model(train_X)  # Apply model to inputs.

        loss = F.mse_loss(yh, train_Y)  # Compute mean squared error between our model output and the correct labels.

        optimizer.zero_grad()  # Set all gradients to 0.
        loss.backward()  # Calculate gradient of loss w.r.t all tensors used to compute loss.
        optimizer.step()  # Update all model parameters according to calculated gradients.

        print(f'epoch {i}, W has value:')
        print(list(model.W.parameters()))
        print('\n')


train_model(model)

# Our model can now be used to predict y values for new X as follows.
new_X = torch.arange(dim).float().view(1, dim)
new_y_h = model(new_X)
print(f'The model predicts {new_y_h}')

# Inspecting Gradients
# Although optimization is handled under-the-hood by pytorch,
# there are some situations where it is helpful to access gradients directly.
# Gradients are stored in each variables .grad attribute.

w = torch.FloatTensor([1, 2])
w.requires_grad = True  # If this flag is set then this variable will store its gradient during backward().
x = torch.FloatTensor([3, 4])
x.requires_grad = True

z = (w * w * x).sum()
z.backward()

print(w.grad)
print(x.grad)


class NeuralNetwork(torch.nn.Module):
    def __init__(self, d_in, d_hit, d_out):
        super().__init__()
        # Define layers here.
        # define linear hidden layer output
        self.hidden = torch.nn.Linear(d_in, d_hit)
        # define linear output layer output
        self.W = torch.nn.Linear(d_hit, d_out)

    def forward(self, x):
        # Compute output here.
        # get hidden layer input
        h_input = self.hidden(x)
        # define activation function for hidden layer
        h_output = torch.relu(h_input)
        # get output layer output
        y_h = self.W(h_output)

        return y_h


model = NeuralNetwork(dim, 4, 1)
train_model(model)


