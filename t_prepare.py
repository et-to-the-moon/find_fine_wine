'''
Prepare Wine Quality Data

Functions:
- wine_out
- split_data
'''

##### IMPORTS #####
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

##### FUNCTIONS #####

def rename_col(df):
    '''Rename columns by replacing spaces with _'''
    # get rid of spaces in column names
    for col in df.columns:
        df = df.rename(columns={col: col.replace(' ','_')})
    return df

def wine_out(df):
    '''Get rid of the small amount of outliers'''
    # Filter rows based on column: 'residual sugar'
    df = df[df['residual_sugar'] < 33]
    # Filter rows based on column: 'free sulfur dioxide'
    df = df[df['free_sulfur_dioxide'] < 280]
    # Filter rows based on column: 'chlorides'
    return df[df['chlorides'] < 0.21]

def add_feature(df):
    '''Add features for explore'''
    # Derive column 'bound_sulfur_dioxide' from column: 'free_sulfur_dioxide'
    df.insert(6, 'bound_sulfur_dioxide', df.apply(lambda row : (row['total_sulfur_dioxide']-row['free_sulfur_dioxide']), axis=1))
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
    df = df[df['free_sulfur_dioxide'] < 280]
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

### SCALERS ###

def mm(train,validate,test,scale=None):
    """
    The function applies the Min Max Scaler method to scale the numerical features of the train, validate,
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
    mm_scale = MinMaxScaler()
    Xtr,Xv,Xt = train[scale],validate[scale],test[scale]
    Xtr = pd.DataFrame(mm_scale.fit_transform(train[scale]),train.index,scale)
    Xv = pd.DataFrame(mm_scale.transform(validate[scale]),validate.index,scale)
    Xt = pd.DataFrame(mm_scale.transform(test[scale]),test.index,scale)
    for col in scale:
        Xtr = Xtr.rename(columns={col: f'{col}_s'})
        Xv = Xv.rename(columns={col: f'{col}_s'})
        Xt = Xt.rename(columns={col: f'{col}_s'})
    return Xtr, Xv, Xt

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

def robs(train,validate,test,scale=None):
    """
    The function applies the RobustScaler method to scale the numerical features of the train, validate,
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
    rob_scale = RobustScaler()
    Xtr,Xv,Xt = train[scale],validate[scale],test[scale]
    Xtr = pd.DataFrame(rob_scale.fit_transform(train[scale]),train.index,scale)
    Xv = pd.DataFrame(rob_scale.transform(validate[scale]),validate.index,scale)
    Xt = pd.DataFrame(rob_scale.transform(test[scale]),test.index,scale)
    for col in scale:
        Xtr = Xtr.rename(columns={col: f'{col}_s'})
        Xv = Xv.rename(columns={col: f'{col}_s'})
        Xt = Xt.rename(columns={col: f'{col}_s'})
    return Xtr, Xv, Xt