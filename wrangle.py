'''
Acquire and Prepare Wine Quality Data

Functions:
- wrangle_wine
    - get_wine
    - rename_col
    - wine_out
- prep_wine
- prep_w_wine
    - wine_out_w
- add_feature
- split_data
- mm
- std
- robs
'''

##### IMPORTS #####
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

##### ACQUIRE #####

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

##### FUNCTIONS #####

def wrangle_wine():
    '''
    Combine acquire and prepare functions
    - get_wine
    - rename_col
    - wine_out
    '''
    df = get_wine()
    df = rename_col(df)
    df = wine_out(df)
    return df

def rename_col(df):
    '''Rename columns
    - replace spaces with _
    - free and total sulfur_dioxide to so2
    '''
    # get rid of spaces in column names
    for col in df.columns:
        df = df.rename(columns={col: col.replace(' ','_')})
    df = df.rename(columns={'free_sulfur_dioxide':'free_so2','total_sulfur_dioxide':'total_so2'})
    return df

def wine_out(df):
    '''Get rid of the small amount of outliers'''
    # Filter rows based on column: 'residual sugar'
    df = df[df['residual_sugar'] < 33]
    # Filter rows based on column: 'free sulfur dioxide'
    df = df[df['free_so2'] < 280]
    # Filter rows based on column: 'chlorides'
    return df[df['chlorides'] < 0.21]

def add_feature(df):
    '''Add features for explore'''
    # Derive column 'bound_sulfur_dioxide' from column: 'free_sulfur_dioxide'
    df.insert(6, 'bound_so2', df.apply(lambda row : (row['total_so2']-row['free_so2']), axis=1))
    # create bool column on sweet or not
    # df.insert(4, 'sweet', df.apply(lambda row : (row['residual_sugar'] > 35), axis=1)) # only one sweet wine so not useful
    return df

def prep_wine(df):
    '''Combined prep functions'''
    df = rename_col(df)
    df = wine_out(df)
    df = add_feature(df)
    return df

def wine_out_w(df):
    '''Get rid of the small amount of outliers ofr white wine'''
    # Filter rows based on column: 'chlorides'
    df = df[df['chlorides'] < .125]
    # Filter rows based on column: 'fixed_acidity'
    df = df[df['fixed_acidity'] < 10.8]
    # Filter rows based on column: 'citric_acid'
    df = df[df['citric_acid'] < 1]
    # Filter rows based on column: 'free_sulfur_dioxide'
    df = df[df['free_so2'] < 280]
    return df[df['residual_sugar'] < 25]

def prep_w_wine(df):
    '''Combined prep functions for white wine'''
    # Drop column: 'red'
    df = df.drop(columns=['red'])
    df = rename_col(df)
    df = wine_out(df)
    df = add_feature(df)
    return df

##### SPLIT DATA #####

def split_data(df):
    '''Split into train, validate, test with a 60/20/20 ratio'''
    train_validate, test = train_test_split(df, test_size=.2, random_state=42)
    train, validate = train_test_split(train_validate, test_size=.25, random_state=42)
    return train, validate, test

### SCALER ###

def std(train,validate,test,scale=None):
    """
    The function applies the Standard Scaler method to scale the numerical features of the train, validate,
    and test datasets.
    
    :param train: a pandas DataFrame containing the training data
    :param validate: The validation dataset, which is used to evaluate the performance of the model
    during training and to tune hyperparameters
    :param test: The "test" parameter is a dataset that is used to evaluate the performance of a machine
    learning model that has been trained on the "train" dataset and validated on the "validate" dataset.
    The "test" dataset is typically used to simulate real-world scenarios and to ensure that the model
    is able
    :return: three dataframes: Xtr (scaled training data), Xv (scaled validation data), and Xt (scaled
    test data).
    """
    if scale is None:
        scale = train.columns.to_list()
    std_scale = StandardScaler()
    Xtr,Xv,Xt = train[scale],validate[scale],test[scale]
    Xtr = pd.DataFrame(std_scale.fit_transform(train[scale]),train.index,scale)
    Xv = pd.DataFrame(std_scale.transform(validate[scale]),validate.index,scale)
    Xt = pd.DataFrame(std_scale.transform(test[scale]),test.index,scale)
    for col in scale:
        Xtr = Xtr.rename(columns={col: f'{col}_s'})
        Xv = Xv.rename(columns={col: f'{col}_s'})
        Xt = Xt.rename(columns={col: f'{col}_s'})
    return Xtr, Xv, Xt
