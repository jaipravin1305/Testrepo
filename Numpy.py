#!/usr/bin/env python
# coding: utf-8

# # Numpy:
# 1. Create a 3x3x3 array with random values

# In[1]:


import numpy as np

# Create a 3x3x3 array with random values between 0 and 1
random_array = np.random.rand(3, 3, 3)

print(random_array)


# # Numpy:
# 2. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal

# In[2]:


matrix = np.zeros((5, 5))

# Set values 1, 2, 3, 4 just below the diagonal
values = [1, 2, 3, 4]
for i in range(1, 5):
    matrix[i, i - 1] = values[i - 1]

print(matrix)


# # Numpy:
# 3. Create a 8x8 matrix and fill it with a checker board pattern

# In[3]:


import numpy as np

# Create an 8x8 matrix filled with zeros
matrix = np.zeros((8, 8))

# Use slicing to set alternate rows and columns to 1
matrix[1::2, ::2] = 1
matrix[::2, 1::2] = 1

print(matrix)


# # Numpy:
# 4. Normalize a 5x5  random matrix

# In[5]:


import numpy as np

# Create a 5x5 random matrix
random_matrix = np.random.rand(5, 5)

# Calculate mean and standard deviation
mean = np.mean(random_matrix)
std_dev = np.std(random_matrix)

# Normalize the matrix
normalized_matrix = (random_matrix - mean) / std_dev

print("Original Random Matrix:")
print(random_matrix)
print("\nNormalized Matrix:")
print(normalized_matrix)


# # Numpy:
# 5. How to find common value between two arrays

# In[4]:


import numpy as np

# Create two NumPy arrays
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([3, 4, 5, 6, 7])

# Find common values between the two arrays
common_values = np.intersect1d(array1, array2)

print("Common values:", common_values)


# # Numpy:
# 6. How to get the dates of yesterday, today, tomorrow.

# In[6]:


import numpy as np
from datetime import datetime, timedelta

# Get the current date (today)
today = datetime.now().date()

# Calculate yesterday and tomorrow
yesterday = today - timedelta(days=1)
tomorrow = today + timedelta(days=1)

# Create NumPy arrays for the dates
dates = np.array([yesterday, today, tomorrow])

# Format the dates as strings if needed
date_strings = [date.strftime("%Y-%m-%d") for date in dates]

print("Yesterday:", date_strings[0])
print("Today:", date_strings[1])
print("Tomorrow:", date_strings[2])


# # Numpy:
# 7. Consider two random array A and B check if they are equal.

# In[7]:


import numpy as np

# Generate two random arrays A and B
A = np.random.rand(5, 5)
B = np.random.rand(5, 5)

# Check if the arrays are equal
if np.array_equal(A, B):
    print("Arrays A and B are equal.")
else:
    print("Arrays A and B are not equal.")


# # Numpy:
# 8. Create a random vector of size 10 and replace the maximum value by 0

# In[11]:


import numpy as np

# Create a random vector of size 10
random_vector = np.random.rand(10)
print(random_vector)
print()
# Find the index of the maximum value in the vector
max_index = np.argmax(random_vector)

# Replace the maximum value with 0
random_vector[max_index] = 0
print("Max Index: ", max_index)
print()
print("Random Vector:")
print(random_vector)


# # Numpy:
# 9. How to print all the values of an array.

# In[ ]:


import numpy as np

# Create a NumPy array
my_array = np.array([1, 2, 3, 4, 5])

# Iterate through the array and print each value
for value in my_array:
    print(value)


# # Numpy:
# 10. Subtract the mean of each row of a matrix

# In[12]:


import numpy as np

# Create a sample matrix (you can replace this with your own matrix)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Calculate the mean of each row
row_means = np.mean(matrix, axis=1, keepdims=True)

# Subtract the row means from the matrix using broadcasting
normalized_matrix = matrix - row_means

print("Original Matrix:")
print(matrix)
print("\nMatrix with Row Means Subtracted:")
print(normalized_matrix)


# # Numpy:
# 11. Consider a given vector how to add 1 to each element indexed by a second vector(be careful with repeated indices)?

# In[13]:


import numpy as np

# Given vector
given_vector = np.array([10, 20, 30, 40, 50])

# Second vector with indices
indices_to_add_1 = np.array([1, 2, 3, 3, 4])

# Find unique indices and their counts
unique_indices, counts = np.unique(indices_to_add_1, return_counts=True)

# Add 1 to each element at the unique indices
given_vector[unique_indices] += counts

print("Given Vector:")
print(given_vector)


# # Numpy:
# 12. How to get a diagonal of dot product?

# In[ ]:


# import numpy as np

# Create two matrices
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

# Compute the dot product of the two matrices
dot_product = np.dot(matrix1, matrix2)

# Extract the diagonal elements using np.diag
diagonal_elements = np.diag(dot_product)

print("Dot Product:")
print(dot_product)

print("\nDiagonal Elements:")
print(diagonal_elements)


# # Numpy:
# 13. How to find the most frequent value in an array?

# In[20]:


import numpy as np

# Create a NumPy array (you can replace this with your own array)
arr = np.random.randint(0,10,50)
print(arr)
# Find the most frequent value
most_frequent_value = np.bincount(arr).argmax()
print()
print("Most frequent value:", most_frequent_value)


# # Numpy:
# 14. How to get the n largest values of an array

# In[22]:


import numpy as np

# Create a NumPy array (you can replace this with your own array)
arr = np.array([5, 2, 8, 1, 9, 3, 7, 4, 6])
print("Original Array: " ,arr)
print()
# Get the n largest values (e.g., n=3)
n = 3
n_largest_values = np.partition(arr, -n)[-n:]

print("N largest values:", n_largest_values)


# # Numpy:
# 15. How to create a record array from a regular array?

# In[23]:


import numpy as np

# Create a regular array
regular_array = np.array([(1, 'Alice', 25),
                          (2, 'Bob', 30),
                          (3, 'Charlie', 22)],
                         dtype=[('ID', int), ('Name', 'U10'), ('Age', int)])

# 'ID', 'Name', and 'Age' are the field names, and their data types are specified.

# Accessing data in the record array
print("ID:", regular_array['ID'])
print("Name:", regular_array['Name'])
print("Age:", regular_array['Age'])


# # Numpy: 
# 16. How to swap two rows of an array?

# In[15]:


import numpy as np

# Create a NumPy array
my_array = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

# Swap rows (for example, swapping row 0 and row 1)
my_array[0, :], my_array[1, :] = my_array[1, :], my_array[0, :]

# Print the modified array
print(my_array)


# # Numpy:
# 17. Write a Python code to reshape to the next dimension of numpy array

# In[16]:


import numpy as np

# Create the original NumPy array
x = np.array([[23, 34, 121], [23, 22, 67], [686, 434, 123]])

# Reshape to the next dimension (1x9)
reshaped_array = x.reshape(1, -1)

# Print the reshaped array
print(reshaped_array)



# In[ ]:




