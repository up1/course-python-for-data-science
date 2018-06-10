import numpy as np

#one dimension
data1 = [1, 2, 3.5, 4, 5]
arr1 = np.array(data1)
print(arr1)
print(type(arr1))
print(arr1.dtype)
print(arr1.ndim)
print(arr1.shape)

#two dimension
data2 = [[1, 2, 3,], [4, 5, 6]]
arr2 = np.array(data2)
print(arr2)
print(arr2.ndim)
print(arr2.shape)

# with numpy library
data = np.zeros(3)
print(data)

data = np.zeros((3,3))
print(data)

data = np.ones(3)
print(data)

data = np.empty(3)
print(data)

# with arange is array-valued like range in python
data = np.arange(10)
print(data)

# create array with specified type
arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)
