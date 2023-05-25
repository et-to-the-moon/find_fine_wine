'''Explore Wine Quality Data

Functions:
'''

########## IMPORTS ##########
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from scipy import stats

########## EXPLORE ##########

def pear(train, x, y, alt_hyp='two-sided'):
    '''Spearman's R test with a print'''
    r,p = stats.spearmanr(train[x], train[y], alternative=alt_hyp)
    print(f"Spearman's R: {x} and {y}\n", f'r = {r}, p = {p}')

def nova(s1,s2,s3):
    '''ANOVA test for 3 samples'''
    stat,p = stats.kruskal(s1,s2,s3)
    print("Kruskal-Wallis H-Test:\n", f'stat = {stat}, p = {p}')

def dist(train):
    '''Wine Quality Distribution'''
    sns.histplot(data=train,x='quality',hue='red',multiple='dodge',discrete=True,palette='coolwarm')
    plt.suptitle('Wine Quality Distribution between Red and White')
    plt.ylabel('# of Wines')
    plt.xlabel('Quality Rating')
    plt.legend(title='Wine',labels=['Red','White'])
    plt.show()