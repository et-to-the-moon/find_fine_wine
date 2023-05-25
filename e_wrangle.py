#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from termcolor import colored
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import env
import numpy as np

def check_file_exists(fn, query, url):
    """
    check if file exists in my local directory, if not, pull from sql db
    return dataframe
    """
    if os.path.isfile(fn):
        print('csv file found and loaded')
        return pd.read_csv(fn, index_col=0)
    else:
        print('creating df and exporting csv')
        df = pd.read_sql(query, url)
        df.to_csv(fn)
        return df
    
def get_zillow_data():
    url = env.get_connection_url('zillow')
    filename = 'zillow.csv'
    query = '''SELECT prop.* ,
            pred.logerror,
            pred.transactiondate,
            air.airconditioningdesc,
            arch.architecturalstyledesc,
            build.buildingclassdesc,
            heat.heatingorsystemdesc,
            land.propertylandusedesc,
            story.storydesc,
            type.typeconstructiondesc
        from properties_2017 prop
        JOIN ( -- used to filter all properties with their last transaction date in 2017, w/o dups
                SELECT parcelid, MAX(transactiondate) AS max_transactiondate
                FROM predictions_2017
                GROUP BY parcelid) trans using (parcelid)
        -- bringing in logerror & transaction_date cols
        JOIN predictions_2017 pred ON trans.parcelid = pred.parcelid
                          AND trans.max_transactiondate = pred.transactiondate
        -- bringing in all other fields related to the properties
        LEFT JOIN airconditioningtype air USING(airconditioningtypeid)
        LEFT JOIN architecturalstyletype arch USING(architecturalstyletypeid)
        LEFT JOIN buildingclasstype build USING(buildingclasstypeid)
        LEFT JOIN heatingorsystemtype heat USING(heatingorsystemtypeid)
        LEFT JOIN propertylandusetype land USING(propertylandusetypeid)
        LEFT JOIN storytype story USING(storytypeid)
        LEFT JOIN typeconstructiontype type USING(typeconstructiontypeid)
        -- exercise stipulations
        WHERE propertylandusedesc = "Single Family Residential"
            AND transactiondate <= '2017-12-31'
            AND prop.longitude IS NOT NULL
            AND prop.latitude IS NOT NULL;'''
    df = check_file_exists(filename, query, url)

    return df

def get_mall_data():
    url = env.get_connection_url('mall_customers')
    filename = 'mall.csv'
    query = ''' SELECT *
             FROM customers
             '''
    df = check_file_exists(filename, query, url)
    
    return df


def get_tips_data():
    url = env.get_connection_url('tips')
    filename = 'tips.csv'
    query = '''select 'total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size' from tips'''
    df = check_file_exists(filename, query, url)

    return df



"""
*-----------------------*
|                       |
|        IMPORTS        |
|                       |
*-----------------------*
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler




"""
*-----------------------*
|                       |
|       FUNCTIONS       |
|                       |
*-----------------------*
"""

def wrangle_exams():
    '''
    read csv from url into df, clean df, and return the prepared df
    '''
    # Read csv file into pandas DataFrame.
    file = "https://gist.githubusercontent.com/o0amandagomez0o/aca6d9c51b425cd9275538db11cb3c60/raw/c22505269e20310abf46df74f9a814a1eddc85c9/student_grades.csv"
    df = pd.read_csv(file)

    #replace blank space with null value
    df.exam3 = df.exam3.replace(' ', np.nan)
    
    #drop all nulls
    df = df.dropna()
    
    #change datatype to exam1 and exam3 to integers
    df.exam1 = df.exam1.astype(int)    
    df.exam3 = df.exam3.astype(int)

    return df


def train_split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123)

    # Take a look at your split datasets

    print(f"""
    train -> {train.shape}
    validate -> {validate.shape}
    test -> {test.shape}""")
    
    return train, validate, test

def eval_dist(r, p, α=0.05):
    if p > α:
        return print(f"""The data is normally distributed""")
    else:
        return print(f"""The data is NOT normally distributed""")
    
    
    
    
def eval_Spearman(r, p, α=0.05):
    if p < α:
        return print(f"""We reject H₀, there is a monotonic relationship.
Spearman's r: {r:2f}
P-value: {p}""")
    else:
        return print(f"""We fail to reject H₀: that there is a monotonic relationship.
Spearman's r: {r:2f}
P-value: {p}""")

    
    
def eval_Pearson(r, p, α=0.05):
    if p < α:
        return print(f"""We reject H₀, there is a linear relationship with a Correlation Coefficient of {r:2f}.
P-value: {p}""")
    else:
        return print(f"""We fail to reject H₀: that there is a linear relationship.
Pearson's r: {r:2f}
P-value: {p}""")
    
    
    

def prep_zillow(df):
    #sum_null = df.isnull().sum()
    df = df.dropna()
    new_columns = {'bedroomcnt': 'Bedrooms', 'bathroomcnt': 'Bathrooms', 'calculatedfinishedsquarefeet': 'SqFt', 'taxvaluedollarcnt': 'Value', 'yearbuilt': 'Yr Built', 'taxamount': 'Tax', 'fips': 'County'}
    df = df.rename(columns=new_columns)
    df['County'] = df['County'].replace({6037: 'LA', 6111: 'Ventura', 6059: 'Orange County'})
    #print(colored(f'Columns with null values: {sum_null[sum_null > 0].index}', 'red'))
    return df


'''def prep_tips(df):
    #sum_null = df.isnull().sum()
    df = df.dropna()
    new_columns = {'total_bill': 'total'}
    dummy_df = pd.get_dummies(data=df[['sex', 'sex', 'smoker', 'day', 'time']]
    #df = pd.concat([df, dummy_df], axis=1)
    return df'''

def train_validate_test_split(X_all, Y):
    '''
    This function takes in a dataframe, the target variable, and a seed for reproducibility.
    It will split the data into train, validate, and test datasets.
    '''

    X_train, X_test, y_train, y_test = train_test_split(X_all, Y, test_size=0.2, random_state=123)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.3, random_state=123)
    
    print(f'''
    X_train -> {X_train.shape}'
    X_validate -> {X_validate.shape}'
    X_test -> {X_test.shape}''') 
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test

def test_train(df, target):

    temp_train, test = train_test_split(df,  test_size= .2, random_state=123, stratify=df[target])
    train, validate = train_test_split(temp_train, test_size= .25, random_state=123, stratify=temp_train[target])
    return train, validate, test

def split_student_data(df):
    X = df.drop(['final_grade'], axis=1)
    Y = df.final_grade
    
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    return X, Y

def split_tips_data(df):
    X = df.drop(['tip'], axis=1)
    Y = df.tip
    
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    return X, Y

def split_swiss_data(df):
    X = df.drop(['Fertility'], axis=1)
    Y = df.Fertility
    
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    return X, Y
    
import numpy as np

def remove_outliers(df, threshold=3):
    """
    Removes outliers from the input data using the z-score method.
    Returns the filtered data without outliers.
    
    Parameters:
    -----------
    data : numpy array
        The input data.
    threshold : float
        The z-score threshold for outlier detection.
    
    Returns:
    --------
    filtered_data : numpy array
        The filtered data without outliers.
    """
    # Calculate the z-scores for each data point
    z_scores = np.abs((df - np.mean(df)) / np.std(df))
    
    # Identify the outliers based on the z-score threshold
    outliers = z_scores > threshold
    
    # Remove the outliers from the data
    filtered_data = df[~outliers]
    
    return df

def test_train(df, target):

    temp_train, test = train_test_split(df,  test_size= .2, random_state=123)
    train, validate = train_test_split(temp_train, test_size= .25, random_state=123)
    return train, validate, test

def split_data(df):
    '''
    Takes in two arguments the dataframe name and the ("name" - must be in string format) to stratify  and
    return train, validate, test subset dataframes will output train, validate, and test in that order
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.25, random_state=123)
    return train, validate, test

def visualize_scaler(scaler, df, columns_to_scale, bins=10):
    #create subplot structure
    fig, axs = plt.subplots(len(columns_to_scale), 2, figsize=(12,12))

    #copy the df for scaling
    df_scaled = df.copy()
    
    #fit and transform the df
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    #plot the pre-scaled data next to the post-scaled data in one row of a subplot
    for (ax1, ax2), col in zip(axs, columns_to_scale):
        ax1.hist(df[col], bins=bins)
        ax1.set(title=f'{col} before scaling', xlabel=col, ylabel='count')
        ax2.hist(df_scaled[col], bins=bins)
        ax2.set(title=f'{col} after scaling with {scaler.__class__.__name__}', xlabel=col, ylabel='count')
    plt.tight_layout()
    plt.show()
    
    