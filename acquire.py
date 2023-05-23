'''
Acquire Wine Quality Data

Functions:

'''

##### IMPORTS #####
import os
import numpy as np
import pandas as pd

##### FUNCTIONS #####

def get_wine():
    """
    This function reads wine data from a CSV file or from data.world and caches it locally for future
    use.
    :return: The function `get_wine()` returns a pandas DataFrame containing information about different
    types of wines. If the data is already cached locally in a CSV file named 'wine.csv', it reads the
    data from the file and returns it. Otherwise, it fetches the data from two different URLs on
    data.world, combines them into a single DataFrame, saves the data to a CSV file named 'wine.csv',
    """
    # filename of csv
    filename='wine.csv'
    # if cached data exist
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    # wrangle from data.world if not cached
    else:
        # data.world links
        df1 = pd.read_csv('https://query.data.world/s/jvqglydhtwbdwm22t6fsontdgvxbuk?dws=00000')
        df1 = df1.assign(red=1)
        df2 = pd.read_csv('https://query.data.world/s/jjdvspurcnyimwd3pp3gnrjifs57fs?dws=00000')
        df2 = df2.assign(red=0)
        df = pd.concat([df1,df2],ignore_index=True)
        # cache data locally
        df.to_csv(filename, index=False)
        return df