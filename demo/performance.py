import numpy as np
import pandas as pd

array = np.arange(1000000)
series = pd.Series(array)

index = np.random.choice(array, size=100)
