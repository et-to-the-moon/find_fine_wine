'''Explore Wine Quality Data

Functions:
'''

########## IMPORTS ##########
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats

import wrangle as w

########## EXPLORE ##########

def pear(train, x, y, alt_hyp='two-sided'):
    '''Spearman's R test with a print'''
    r,p = stats.spearmanr(train[x], train[y], alternative=alt_hyp)
    print(f"Spearman's R: {x} and {y}\n", f'r = {r}, p = {p}')

def nova3(s1,s2,s3):
    '''ANOVA test for 3 samples'''
    stat,p = stats.kruskal(s1,s2,s3)
    print("Kruskal-Wallis H-Test:\n", f'stat = {stat}, p = {p}')

def nova5(s1,s2,s3,s4,s5):
    '''ANOVA test for 3 samples'''
    stat,p = stats.kruskal(s1,s2,s3,s4,s5)
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
    boxen = plt.title('Quality among Red and White Wines')
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

def plot_violin(train, target, quant_var, loc='upper center', swap=False):
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
    if swap==True:
        p = sns.violinplot(data=train, y=target, x=quant_var, hue=target, dodge=False, color='red')
        p = plt.axvline(average, ls='--', color='black')
    else:
        p = sns.violinplot(data=train, x=target, y=quant_var, hue=target, dodge=False, color='red')
        p = plt.axhline(average, ls='--', color='black')
    p = plt.legend(loc=loc)
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

##### CLUSTERS #####

def cluck2(df_s,df_t,f,clusters):
    '''Cluster check 2 features'''
    X = df_s[f]
    km = KMeans(n_clusters=clusters,random_state=42)
    km.fit(X)
    df_t[f'cluster_{f[0][0]}{f[1][0]}{clusters}'] = km.predict(X)
    df_t.quality = df_t.quality.astype(str)
    df_t[f'cluster_{f[0][0]}{f[1][0]}{clusters}'] = df_t[f'cluster_{f[0][0]}{f[1][0]}{clusters}'].astype(str)
    plt.figure(figsize=[16,4])
    plt.subplot(121)
    sns.scatterplot(data=df_t, x=f[0][:-2], y=f[1][:-2], alpha=.6, hue='quality')
    plt.subplot(122)
    sns.scatterplot(data=df_t, x=f[0][:-2], y=f[1][:-2], alpha=.6, hue=f'cluster_{f[0][0]}{f[1][0]}{clusters}')
    plt.show()

def vol_sug_red_explore(train):
    '''Cluster check 2 features'''
    # create cluster df
    vsr = train[train.wine_type=='red'][['volatile_acidity','residual_sugar','quality']]
    # scale it
    X = pd.DataFrame(StandardScaler().fit_transform(vsr[['volatile_acidity','residual_sugar']]),vsr.index,['volatile_acidity','residual_sugar'])
    # cluster
    km = KMeans(n_clusters=3,random_state=42)
    km.fit(X)
    # back to df
    vsr['volatile_sugar'] = km.predict(X)
    # vsr.quality = vsr.quality.astype(str)
    # stats test
    nova3(vsr[vsr.volatile_sugar==0].quality,vsr[vsr.volatile_sugar==1].quality,vsr[vsr.volatile_sugar==2].quality)
    # plot it
    vsr.volatile_sugar = vsr.volatile_sugar.map({0:'hi_acid_low_sug',1:'low_acid_low_sug',2:'med_acid_hi_sug'})
    vio_vsr = plot_violin(vsr,'volatile_sugar','quality',loc='lower left')
    vio_vsr = plt.title('Volatile Sugar Cluster Quality Check')
    plt.show()

def vol_sug_red_cluster(Xtr,Xv):
    '''Create red wine 'volatile_acidity_s','residual_sugar_s' cluster and scale for modeling
    
    This will scale, cluster, add cluster to unscaled, then scale for modeling'''
    Xtr_s,Xv_s,dummy_X = w.std(Xtr,Xv,Xv)
    Xtr_svs,Xv_svs = Xtr_s[['volatile_acidity_s_s','residual_sugar_s_s']],Xv_s[['volatile_acidity_s_s','residual_sugar_s_s']]
    km = KMeans(n_clusters=3,random_state=42)
    km.fit(Xtr_svs)
    Xtr['volatile_sugar'],Xv['volatile_sugar'] = km.predict(Xtr_svs),km.predict(Xv_svs)
    Xtr.volatile_sugar = Xtr.volatile_sugar.map({0:'hi_acid_low_sug',1:'low_acid_low_sug',2:'med_acid_hi_sug'})
    Xv.volatile_sugar = Xv.volatile_sugar.map({0:'hi_acid_low_sug',1:'low_acid_low_sug',2:'med_acid_hi_sug'})
    Xtr,Xv = pd.concat([Xtr,pd.get_dummies(Xtr.volatile_sugar)],axis=1),pd.concat([Xv,pd.get_dummies(Xv.volatile_sugar)],axis=1)
    Xtr_s,Xv_s,dummy_s = w.std(Xtr.select_dtypes(exclude='object'),Xv.select_dtypes(exclude='object'),Xv.select_dtypes(exclude='object'))
    return Xtr_s,Xv_s

def vol_sug_white_explore(train):
    '''Cluster check 2 features'''
    # create cluster df
    vsw = train[train.wine_type=='white'][['volatile_acidity','residual_sugar','quality']]
    # scale it
    X = pd.DataFrame(StandardScaler().fit_transform(vsw[['volatile_acidity','residual_sugar']]),vsw.index,['volatile_acidity','residual_sugar'])
    # cluster
    km = KMeans(n_clusters=3,random_state=42)
    km.fit(X)
    # back to df
    vsw['volatile_sugar'] = km.predict(X)
    # vsw.quality = vsw.quality.astype(str)
    # stats test
    nova3(vsw[vsw.volatile_sugar==0].quality,vsw[vsw.volatile_sugar==1].quality,vsw[vsw.volatile_sugar==2].quality)
    # plot it
    vsw.volatile_sugar = vsw.volatile_sugar.map({0:'low_acid_hi_sug',1:'hi_acid_low_sug',2:'low_acid_low_sug'})
    vio_vsw = plot_violin(vsw,'volatile_sugar','quality',loc='lower left')
    vio_vsw = plt.title('Volatile Sugar Cluster Quality Check')
    plt.show()

def vol_sug_white_cluster(Xtr,Xv):
    '''Create white wine 'volatile_acidity_s','residual_sugar_s' cluster and scale for modeling
    
    This will scale, cluster, add cluster to unscaled, then scale for modeling'''
    Xtr_s,Xv_s,dummy_X = w.std(Xtr,Xv,Xv)
    Xtr_svs,Xv_svs = Xtr_s[['volatile_acidity_s_s','residual_sugar_s_s']],Xv_s[['volatile_acidity_s_s','residual_sugar_s_s']]
    km = KMeans(n_clusters=3,random_state=42)
    km.fit(Xtr_svs)
    Xtr['volatile_sugar'],Xv['volatile_sugar'] = km.predict(Xtr_svs),km.predict(Xv_svs)
    Xtr.volatile_sugar = Xtr.volatile_sugar.map({0:'low_acid_hi_sug',1:'hi_acid_low_sug',2:'low_acid_low_sug'})
    Xv.volatile_sugar = Xv.volatile_sugar.map({0:'low_acid_hi_sug',1:'hi_acid_low_sug',2:'low_acid_low_sug'})
    Xtr,Xv = pd.concat([Xtr,pd.get_dummies(Xtr.volatile_sugar)],axis=1),pd.concat([Xv,pd.get_dummies(Xv.volatile_sugar)],axis=1)
    Xtr_s,Xv_s,dummy_s = w.std(Xtr.select_dtypes(exclude='object'),Xv.select_dtypes(exclude='object'),Xv.select_dtypes(exclude='object'))
    return Xtr_s,Xv_s

def den_alc_white_explore(train):
    '''Cluster check 2 features'''
    # create cluster df
    daw = train[train.wine_type=='white'][['density','alcohol','quality']]
    # scale it
    X = pd.DataFrame(StandardScaler().fit_transform(daw[['density','alcohol']]),daw.index,['density','alcohol'])
    # cluster
    km = KMeans(n_clusters=5,random_state=42)
    km.fit(X)
    # back to df
    daw['density_alcohol'] = km.predict(X)
    # daw.quality = daw.quality.astype(str)
    # stats test
    nova5(daw[daw.density_alcohol==0].quality,daw[daw.density_alcohol==1].quality,daw[daw.density_alcohol==2].quality,daw[daw.density_alcohol==3].quality,daw[daw.density_alcohol==4].quality)
    # plot it
    daw.density_alcohol = daw.density_alcohol.map({0:'hi_den_low_alc',1:'low_den_med_alc',2:'med_den_med_alc',3:'low_den_hi_alc',4:'med_den_low_alc'})
    vio_daw = plot_violin(daw,'density_alcohol','quality',loc='center right', swap=True)
    vio_daw = plt.title('Density Alcohol Cluster Quality Check')
    plt.show()

def den_alc_white_cluster(Xtr,Xv):
    '''Create white wine density alcohol cluster and scale for modeling
    
    This will scale, cluster, add cluster to unscaled, then scale for modeling'''
    Xtr_s,Xv_s,dummy_X = w.std(Xtr,Xv,Xv)
    Xtr_sda,Xv_sda = Xtr_s[['density_s_s','alcohol_s_s']],Xv_s[['density_s_s','alcohol_s_s']]
    km = KMeans(n_clusters=5,random_state=42)
    km.fit(Xtr_sda)
    Xtr['density_alcohol'],Xv['density_alcohol'] = km.predict(Xtr_sda),km.predict(Xv_sda)
    Xtr.density_alcohol = Xtr.density_alcohol.map({0:'hi_den_low_alc',1:'low_den_med_alc',2:'med_den_med_alc',3:'low_den_hi_alc',4:'med_den_low_alc'})
    Xv.density_alcohol = Xv.density_alcohol.map({0:'hi_den_low_alc',1:'low_den_med_alc',2:'med_den_med_alc',3:'low_den_hi_alc',4:'med_den_low_alc'})
    Xtr,Xv = pd.concat([Xtr,pd.get_dummies(Xtr.density_alcohol)],axis=1),pd.concat([Xv,pd.get_dummies(Xv.density_alcohol)],axis=1)
    Xtr_s,Xv_s,dummy_s = w.std(Xtr.select_dtypes(exclude='object'),Xv.select_dtypes(exclude='object'),Xv.select_dtypes(exclude='object'))
    return Xtr_s,Xv_s

def den_alc_red_explore(train):
    '''Cluster check 2 features'''
    # create cluster df
    dar = train[train.wine_type=='red'][['density','alcohol','quality']]
    # scale it
    X = pd.DataFrame(StandardScaler().fit_transform(dar[['density','alcohol']]),dar.index,['density','alcohol'])
    # cluster
    km = KMeans(n_clusters=5,random_state=42)
    km.fit(X)
    # back to df
    dar['density_alcohol'] = km.predict(X)
    # dar.quality = dar.quality.astype(str)
    # stats test
    nova5(dar[dar.density_alcohol==0].quality,dar[dar.density_alcohol==1].quality,dar[dar.density_alcohol==2].quality,dar[dar.density_alcohol==3].quality,dar[dar.density_alcohol==4].quality)
    # plot it
    dar.density_alcohol = dar.density_alcohol.map({0:'hi_den_low_alc',1:'low_den_med_alc',2:'med_den_med_alc',3:'low_den_hi_alc',4:'med_den_low_alc'})
    vio_dar = plot_violin(dar,'density_alcohol','quality',loc='upper left', swap=True)
    vio_dar = plt.title('Density Alcohol Cluster Quality Check')
    plt.show()

def den_alc_red_cluster(Xtr,Xv):
    '''Create red wine density alcohol cluster and scale for modeling
    
    This will scale, cluster, add cluster to unscaled, then scale for modeling'''
    Xtr_s,Xv_s,dummy_X = w.std(Xtr,Xv,Xv)
    Xtr_sda,Xv_sda = Xtr_s[['density_s_s','alcohol_s_s']],Xv_s[['density_s_s','alcohol_s_s']]
    km = KMeans(n_clusters=5,random_state=42)
    km.fit(Xtr_sda)
    Xtr['density_alcohol'],Xv['density_alcohol'] = km.predict(Xtr_sda),km.predict(Xv_sda)
    Xtr.density_alcohol = Xtr.density_alcohol.map({0:'med_den_med_alc',1:'hi_den_low_alc',2:'med_den_low_alc',3:'hi_den_med_alc',4:'low_den_hi_alc'})
    Xv.density_alcohol = Xv.density_alcohol.map({0:'med_den_med_alc',1:'hi_den_low_alc',2:'med_den_low_alc',3:'hi_den_med_alc',4:'low_den_hi_alc'})
    Xtr,Xv = pd.concat([Xtr,pd.get_dummies(Xtr.density_alcohol)],axis=1),pd.concat([Xv,pd.get_dummies(Xv.density_alcohol)],axis=1)
    Xtr_s,Xv_s,dummy_s = w.std(Xtr.select_dtypes(exclude='object'),Xv.select_dtypes(exclude='object'),Xv.select_dtypes(exclude='object'))
    return Xtr_s,Xv_s