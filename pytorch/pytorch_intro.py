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



