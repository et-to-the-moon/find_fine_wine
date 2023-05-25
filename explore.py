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
    sns.histplot(data=train,x='quality',hue='wine_type',multiple='dodge',discrete=True,palette='coolwarm')
    plt.suptitle('Wine Quality Distribution between Red and White')
    plt.ylabel('# of Wines')
    plt.xlabel('Quality Rating')
    plt.legend(title='Wine',labels=['Red','White'])
    plt.show()

def type_quality(train,target,quant_var):
    """
    The function explores the relationship between a quality and wine type using
    descriptive statistics, a boxen plot, a swarm plot, and a Mann-Whitney test.
    
    :param train: a pandas DataFrame containing the training data
    :param target: The target variable is the variable that we are trying to predict or understand in
    our analysis. It is the dependent variable in a regression or classification problem. In this
    function, it is used to group the data and compare the descriptive statistics and means between
    different groups
    :param quant_var: The quantitative variable that we want to explore in relation to the target
    variable
    """
    print(quant_var, "\n____________________")
    # descriptive_stats = train.groupby(target)[quant_var].describe()
    average = train[quant_var].mean()
    compare_means(train, target, quant_var)
    # print(descriptive_stats, "\n")
    # print("\nMann-Whitney Test:\n", f'stat = {stat}, p = {p}')
    print("____________________")
    # plt.figure(figsize=(4,4))
    boxen = plot_violin(train, target, quant_var)
    # swarm = plot_swarm(train, target, quant_var)
    plt.show()

def explore_bivariate_quant(train, target, quant_var):
    """
    The function explores the relationship between a quantitative variable and a target variable using
    descriptive statistics, a boxen plot, a swarm plot, and a Mann-Whitney test.
    
    :param train: a pandas DataFrame containing the training data
    :param target: The target variable is the variable that we are trying to predict or understand in
    our analysis. It is the dependent variable in a regression or classification problem. In this
    function, it is used to group the data and compare the descriptive statistics and means between
    different groups
    :param quant_var: The quantitative variable that we want to explore in relation to the target
    variable
    """
    print(quant_var, "\n____________________")
    # descriptive_stats = train.groupby(target)[quant_var].describe()
    average = train[quant_var].mean()
    compare_means(train, target, quant_var)
    # print(descriptive_stats, "\n")
    # print("\nMann-Whitney Test:\n", f'stat = {stat}, p = {p}')
    print("____________________")
    plt.figure(figsize=(4,4))
    boxen = plot_boxen(train, target, quant_var)
    swarm = plot_swarm(train, target, quant_var)
    plt.show()

def compare_means(train, target, quant_var, alt_hyp='two-sided'):
    """
    The function compares the means of two groups using the Mann-Whitney U test and returns the test
    statistic and p-value.
    
    :param train: a pandas DataFrame containing the training data
    :param target: The target variable is a binary variable that indicates the outcome of interest. In
    this function, it is used to split the data into two groups based on the value of the target
    variable (0 or 1)
    :param quant_var: The quantitative variable that we want to compare the means of between two groups
    :param alt_hyp: The alternative hypothesis for the Mann-Whitney U test. It specifies the direction
    of the test and can be either "two-sided" (default), "less" or "greater". "two-sided" means that the
    test is two-tailed, "less" means that the test is one, defaults to two-sided (optional)
    :return: the result of a Mann-Whitney U test comparing the means of two groups (x and y) based on a
    quantitative variable (quant_var) in a training dataset (train) with a binary target variable
    (target). The alternative hypothesis (alt_hyp) can be specified as either 'two-sided' (default),
    'less', or 'greater'.
    """
    x = train[train[target]=='red'][quant_var]
    y = train[train[target]=='white'][quant_var]
    stat,p = stats.mannwhitneyu(x, y, use_continuity=True, alternative=alt_hyp)
    print("Mann-Whitney Test:\n", f'stat = {stat}, p = {p}')

def plot_boxen(train, target, quant_var):
    """
    This function plots a boxenplot with a horizontal line representing the mean of a quantitative
    variable for each category of a target variable in a given dataset.
    
    :param train: a pandas DataFrame containing the training data
    :param target: The target variable is the categorical variable that we want to compare the
    distribution of the quantitative variable across. For example, if we are analyzing the relationship
    between income and education level, the target variable would be education level (e.g. high school,
    college, graduate degree)
    :param quant_var: The quantitative variable that we want to plot on the y-axis of the boxen plot
    :return: a boxenplot with a horizontal line representing the mean value of the quantitative
    variable, and a title indicating the name of the variable being plotted. The variable `p` is being
    returned, which is the result of the last plotted object (in this case, the title). However, it is
    not necessary to return `p` since it is not being used outside of the function
    """
    average = train[quant_var].mean()
    p = sns.boxenplot(data=train, x=target, y=quant_var, hue=target, color='red')
    p = plt.title(quant_var)
    p = plt.legend(loc='upper center')
    p = plt.axhline(average, ls='--', color='black')
    return p

def plot_violin(train, target, quant_var):
    """
    This function plots a boxenplot with a horizontal line representing the mean of a quantitative
    variable for each category of a target variable in a given dataset.
    
    :param train: a pandas DataFrame containing the training data
    :param target: The target variable is the categorical variable that we want to compare the
    distribution of the quantitative variable across. For example, if we are analyzing the relationship
    between income and education level, the target variable would be education level (e.g. high school,
    college, graduate degree)
    :param quant_var: The quantitative variable that we want to plot on the y-axis of the boxen plot
    :return: a boxenplot with a horizontal line representing the mean value of the quantitative
    variable, and a title indicating the name of the variable being plotted. The variable `p` is being
    returned, which is the result of the last plotted object (in this case, the title). However, it is
    not necessary to return `p` since it is not being used outside of the function
    """
    average = train[quant_var].mean()
    # p = plt.figure(figsize=[4,4])
    p = sns.violinplot(data=train, x=target, y=quant_var, hue=target, dodge=False, color='red')
    p = plt.title('Quality among Red and White Wines')
    p = plt.legend(loc='upper center')
    p = plt.xlabel('Wine Type')
    p = plt.ylabel('Wine Quality')
    p = plt.axhline(average, ls='--', color='black')
    return p

def plot_swarm(train, target, quant_var,size=3):
    """
    The function plots a swarmplot with a horizontal line indicating the mean value of a quantitative
    variable for a given target variable.
    
    :param train: a pandas DataFrame containing the training data
    :param target: The target variable is the variable that we want to predict or explain using the
    other variables in the dataset. It is usually represented on the x-axis of a plot
    :param quant_var: The quantitative variable that we want to plot on the y-axis of the swarm plot
    :return: the plot object `p`.
    """
    average = train[quant_var].mean()
    p = sns.swarmplot(data=train, x=target, y=quant_var, size=size, marker='.', color='black')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p

def eval_dist(r, p, α=0.05):
    if p > α:
        return print("""The data is normally distributed""")
    else:
        return print("""The data is NOT normally distributed""")