import numpy as np

a = np.random.rand(1000, 1000)
b = np.dot(a, a)

from threadpoolctl import threadpool_info

info = threadpool_info()
for entry in info:
    print(entry)