import numpy as np
from timeit import default_timer as timer

start = timer()
my_arr = np.arange(1000000)
for _ in range(10): my_arr2 = my_arr * 2
end = timer()
print(end - start)
