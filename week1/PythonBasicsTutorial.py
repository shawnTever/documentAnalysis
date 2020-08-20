my_list2 = [i * i for i in range(10)]  # Creates a list of the first 10 square integers.
my_set2 = {i for i in range(10)}
my_dict2 = {i: i * 3 + 1 for i in range(10)}

print(my_list2)
print(my_set2)
print(my_dict2)

# Comprehensions can also range over the elements in a data structure.
my_list3 = [my_list2[i] * i for i in my_set2]
print(my_list3)

dictionary = {}
for a in my_list2:
    if a in my_set2:
        dictionary[a] = True
    else:
        dictionary[a] = False
print(dictionary)

import numpy as np  # This statement imports the package numpy and gives it the name np

my_array1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # This is a 2x3 matrix.
my_array2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # This is a 3x2 matrix.

# elements addition
np.sum(my_array1)
# Matrix multiplication
np.matmul(my_array1, my_array2)
# Matrix transpose
print(np.transpose(my_array2))
# Operations such as '+' and '*' can be directly applied to matrices
my_array1 + np.transpose(my_array2)
# The element in the first row and second column.
print(my_array1[0, 1])
# second row.
print(my_array1[1, :])
# Everything from the second row onwards.
print(my_array1[1:])
# second column.
print(my_array1[:, 1])
# Everything from the second column onwards.
print(my_array1[:, 1:])
# Everything upto (but not including) the last row
print(my_array1[:-1])
# Everything upto (but not including) the last column
print(my_array1[:, :-1])
# computing the sum of all of the elements in my_array1 except for the last column.
print(np.sum(my_array1[:, :-1]))

import pandas as pd

my_df = pd.DataFrame({'c1': [1.0, 2.0, 3.0],
                      'c2': ['a', 'b', 'c'],
                      'c3': [True, False, True]})

print(my_df)
print(my_df.shape)

print(my_df.iloc[1:, :2])  # From the second row on, up to the second column
# index a DataFrame by using column names
print(my_df['c2'])

