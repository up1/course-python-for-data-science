import pandas as pd
import re

def get_title(name):
    title = re.search(' ([A-Za-z]+)\.', name)
    if title:
        return title.group(1)
    return ""
    
if __name__== "__main__":
    titanic_df = pd.read_csv("train.csv")
    titanic_df['Title'] = titanic_df['Name'].apply(get_title)
    print(pd.crosstab(titanic_df['Title'], titanic_df['Sex']))
