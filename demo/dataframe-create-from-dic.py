import pandas as pd

data = {
 'weekday': ['Sun', 'Sun', 'Mon', 'Mon'],
 'city': ['Bangkok', 'CNX', 'Bangkok', 'CNX'],
 'visitors': [1000, None, 500, 1500]
}

users = pd.DataFrame(data)
print(users)
print(users.info())
