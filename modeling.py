'''Model Zillow data

Functions:
- metrics_reg
- baseline
- rfe_rev
- reg_mods
- final_models
- cluster_model
- test_model
- plt_err
'''

########## IMPORTS ##########
import pandas as pd
import numpy as np
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE

import wrangle as w
import explore as e

######### FUNCTIONS #########

def metrics_reg(y, y_pred):
    """
    Input y and y_pred & get RMSE, R2
    """
    rmse = mean_squared_error(y, y_pred, squared=False)
    r2 = r2_score(y, y_pred)
    return round(rmse,2), round(r2,4)

def baseline(train,val):
    """
    The function calculates and prints the baseline metrics of a model
    that always predicts the mean in the target variable.
    """
    blt = train.copy()
    blv = val.copy()
    pred_mean = blt[['quality']].mean()[0]
    ytr_p = blt.assign(pred_mean=pred_mean)
    yv_p = blv.assign(pred_mean=pred_mean)
    rmse_tr = mean_squared_error(blt.quality,ytr_p.pred_mean)**.5
    rmse_v = mean_squared_error(blv.quality,yv_p.pred_mean)**.5
    print('                  Red          White')
    print(f'Baseline    Mean: {round(((blt[blt.wine_type=="red"].quality).mean()),2)}   Mean: {round(((blt[blt.wine_type=="white"].quality).mean()),2)}')
    print(f'Train       RMSE: {round(rmse_tr,2)}  RMSE:  {round(rmse_tr,2)}')
    print(f'Validate    RMSE: {round(rmse_v,2)}  RMSE:  {round(rmse_v,2)}')

def rfe_rev(Xs_train,y_train,r):
    '''Get RFE ranks in a dataframe'''
    lr = LinearRegression()
    rfe = RFE(lr,n_features_to_select=r)
    rfe.fit(Xs_train,y_train)
    rfe_ranks_df = pd.DataFrame({'Var':Xs_train.columns.to_list(),'Rank':rfe.ranking_})
    return rfe_ranks_df.sort_values('Rank')

def reg_mods(Xtr,ytr,Xv,yv,features=None,alpha=1,degree=2):
    '''
    Input scaled X_train,y_train,X_val,y_val, list of features, alpha, and degree
    so that function will run through linear regression, lasso lars, and
    polynomial feature regression
    - diff feature combos
    - diff hyper params
    - output as df
    '''
    if features is None:
        features = Xtr.columns.to_list()
    # baseline as mean
    pred_mean = ytr.mean()[0]
    ytr_p = ytr.assign(pred_mean=pred_mean)
    yv_p = yv.assign(pred_mean=pred_mean)
    rmse_tr = mean_squared_error(ytr,ytr_p.pred_mean)**.5
    rmse_v = mean_squared_error(yv,yv_p.pred_mean)**.5
    r2_tr = r2_score(ytr, ytr_p.pred_mean)
    r2_v = r2_score(yv, yv_p.pred_mean)
    output = {
            'model':'bl_mean',
            'features':'None',
            'params':'None',
            'rmse_tr':rmse_tr,
            'rmse_v':rmse_v,
            'r2_tr':r2_tr,
            'r2_v':r2_v
        }
    metrics = [output]
    # create iterable for feature combos
    for r in range(1,(len(features)+1)):
        # cycle through feature combos for linear reg
        for feature in itertools.combinations(features,r):
            f = list(feature)
            # linear regression
            lr = LinearRegression()
            lr.fit(Xtr[f],ytr)
            # metrics
            pred_lr_tr = lr.predict(Xtr[f])
            rmse_tr,r2_tr = metrics_reg(ytr,pred_lr_tr)
            pred_lr_v = lr.predict(Xv[f])
            rmse_v,r2_v = metrics_reg(yv,pred_lr_v)
            # table-ize
            output ={
                    'model':'LinearRegression',
                    'features':f,
                    'params':'None',
                    'rmse_tr':rmse_tr,
                    'rmse_v':rmse_v,
                    'r2_tr':r2_tr,
                    'r2_v':r2_v
                }
            metrics.append(output)
        # cycle through feature combos and alphas for lasso lars # poor performance with this data, worse than baseline mean
        # for feature,a in itertools.product(itertools.combinations(features,r),alpha):
        #     f = list(feature)
        #     # lasso lars
        #     ll = LassoLars(alpha=a,normalize=False,random_state=42)
        #     ll.fit(Xtr[f],ytr)
        #     # metrics
        #     pred_ll_tr = ll.predict(Xtr[f])
        #     rmse_tr,r2_tr = metrics_reg(ytr,pred_ll_tr)
        #     pred_ll_v = ll.predict(Xv[f])
        #     rmse_v,r2_v = metrics_reg(yv,pred_ll_v)
        #     # table-ize
        #     output ={
        #             'model':'LassoLars',
        #             'features':f,
        #             'params':f'alpha={a}',
        #             'rmse_tr':rmse_tr,
        #             'rmse_v':rmse_v,
        #             'r2_tr':r2_tr,
        #             'r2_v':r2_v
        #         }
        #     metrics.append(output)
        # cycle through feature combos and degrees for polynomial feature reg
        for feature,d in itertools.product(itertools.combinations(features,r),degree):
            f = list(feature)
            # polynomial feature regression
            pf = PolynomialFeatures(degree=d)
            Xtr_pf = pf.fit_transform(Xtr[f])
            Xv_pf = pf.transform(Xv[f])
            lp = LinearRegression()
            lp.fit(Xtr_pf,ytr)
            # metrics
            pred_lp_tr = lp.predict(Xtr_pf)
            rmse_tr,r2_tr = metrics_reg(ytr,pred_lp_tr)
            pred_lp_v = lp.predict(Xv_pf)
            rmse_v,r2_v = metrics_reg(yv,pred_lp_v)
            # table-ize
            output ={
                    'model':'PolynomialFeature',
                    'features':f,
                    'params':f'degree={d}',
                    'rmse_tr':rmse_tr,
                    'rmse_v':rmse_v,
                    'r2_tr':r2_tr,
                    'r2_v':r2_v
                }
            metrics.append(output)
        # cycle through feature combos, alphas, and powers for tweedie reg
        for feature,a in itertools.product(itertools.combinations(features,r),alpha):
            f = list(feature)
            # tweedie regressor glm
            lm = TweedieRegressor(power=0,alpha=a)
            lm.fit(Xtr[f],ytr.quality)
            # metrics
            pred_lm_tr = lm.predict(Xtr[f])
            rmse_tr,r2_tr = metrics_reg(ytr,pred_lm_tr)
            pred_lm_v = lm.predict(Xv[f])
            rmse_v,r2_v = metrics_reg(yv,pred_lm_v)
            # table-ize
            output ={
                    'model':'TweedieRegressor',
                    'features':f,
                    'params':f'power=0,alpha={a}',
                    'rmse_tr':rmse_tr,
                    'rmse_v':rmse_v,
                    'r2_tr':r2_tr,
                    'r2_v':r2_v
                }
            metrics.append(output)
    return pd.DataFrame(metrics)

def final_models(model,Xr_train,Xw_train,yr_train,yw_train,Xr_val,Xw_val,yr_val,yw_val):
    '''Input model type along with scaled train and validate data and
    it will return RMSE results per the selected model
    
    Please include model argument: lr, poly, tweedie, lasso'''
    if model == 'lr':
        # features
        fr=['fixed_acidity_s', 'volatile_acidity_s', 'residual_sugar_s', 'chlorides_s', 'total_so2_s', 'density_s', 'sulphates_s', 'alcohol_s']
        fw=['volatile_acidity_s', 'residual_sugar_s', 'chlorides_s', 'free_so2_s', 'total_so2_s', 'pH_s', 'sulphates_s', 'alcohol_s']
        # model
        lrr = LinearRegression()
        lrw = LinearRegression()
        lrr.fit(Xr_train[fr],yr_train)
        lrw.fit(Xw_train[fw],yw_train)
        # metrics red
        pred_lrr_tr = lrr.predict(Xr_train[fr])
        rmse_trr,r2_tr = metrics_reg(yr_train,pred_lrr_tr)
        pred_lrr_v = lrr.predict(Xr_val[fr])
        rmse_vr,r2_v = metrics_reg(yr_val,pred_lrr_v)
        # metrics white
        pred_lrw_tr = lrw.predict(Xw_train[fw])
        rmse_trw,r2_tr = metrics_reg(yw_train,pred_lrw_tr)
        pred_lrw_v = lrw.predict(Xw_val[fw])
        rmse_vw,r2_v = metrics_reg(yw_val,pred_lrw_v)
        print('Linear Reg        Red          White')
        print(f'Train       RMSE: {round(rmse_trr,2)}  RMSE:  {round(rmse_trw,2)}')
        print(f'Validate    RMSE: {round(rmse_vr,2)}  RMSE:  {round(rmse_vw,2)}')
    elif model == 'poly':
        # features
        fr=['volatile_acidity_s', 'total_so2_s', 'sulphates_s', 'alcohol_s']
        fw=['fixed_acidity_s', 'volatile_acidity_s', 'citric_acid_s', 'chlorides_s', 'free_so2_s', 'total_so2_s', 'alcohol_s']
        # polynomial feature regression
        pfr = PolynomialFeatures(degree=3)
        pfw = PolynomialFeatures(degree=3)
        Xr_train_pf = pfr.fit_transform(Xr_train[fr])
        Xr_val_pf = pfr.transform(Xr_val[fr])
        Xw_train_pf = pfw.fit_transform(Xw_train[fw])
        Xw_val_pf = pfw.transform(Xw_val[fw])
        # model
        prr = LinearRegression()
        prw = LinearRegression()
        prr.fit(Xr_train_pf,yr_train)
        prw.fit(Xw_train_pf,yw_train)
        # metrics red
        pred_prr_tr = prr.predict(Xr_train_pf)
        rmse_trr,r2_tr = metrics_reg(yr_train,pred_prr_tr)
        pred_prr_v = prr.predict(Xr_val_pf)
        rmse_vr,r2_v = metrics_reg(yr_val,pred_prr_v)
        # metrics white
        pred_prw_tr = prw.predict(Xw_train_pf)
        rmse_trw,r2_tr = metrics_reg(yw_train,pred_prw_tr)
        pred_prw_v = prw.predict(Xw_val_pf)
        rmse_vw,r2_v = metrics_reg(yw_val,pred_prw_v)
        print('Polynomial        Red          White')
        print(f'Train       RMSE: {round(rmse_trr,2)}  RMSE:  {round(rmse_trw,2)}')
        print(f'Validate    RMSE: {round(rmse_vr,2)}  RMSE:  {round(rmse_vw,2)}')
    elif model == 'tweedie':
        # features
        fr=['fixed_acidity_s', 'volatile_acidity_s', 'citric_acid_s', 'free_so2_s', 'total_so2_s', 'pH_s', 'sulphates_s', 'alcohol_s']
        fw=['volatile_acidity_s', 'citric_acid_s', 'residual_sugar_s', 'chlorides_s', 'free_so2_s', 'total_so2_s', 'density_s', 'pH_s', 'sulphates_s', 'alcohol_s']
        # model
        trr = TweedieRegressor(alpha=1,power=0)
        trw = TweedieRegressor(alpha=1,power=0)
        trr.fit(Xr_train[fr],yr_train.quality)
        trw.fit(Xw_train[fw],yw_train.quality)
        # metrics red
        pred_trr_tr = trr.predict(Xr_train[fr])
        rmse_trr,r2_tr = metrics_reg(yr_train,pred_trr_tr)
        pred_trr_v = trr.predict(Xr_val[fr])
        rmse_vr,r2_v = metrics_reg(yr_val,pred_trr_v)
        # metrics white
        pred_trw_tr = trw.predict(Xw_train[fw])
        rmse_trw,r2_tr = metrics_reg(yw_train,pred_trw_tr)
        pred_trw_v = trw.predict(Xw_val[fw])
        rmse_vw,r2_v = metrics_reg(yw_val,pred_trw_v)
        print('Tweedie           Red          White')
        print(f'Train       RMSE: {round(rmse_trr,2)}  RMSE:  {round(rmse_trw,2)}')
        print(f'Validate    RMSE: {round(rmse_vr,2)}  RMSE:  {round(rmse_vw,2)}')
    elif model == 'lasso':
        # features
        fr=Xr_train.columns
        fw=Xw_train.columns
        # model
        llr = LassoLars(alpha=1,normalize=False,random_state=42)
        llw = LassoLars(alpha=1,normalize=False,random_state=42)
        llr.fit(Xr_train[fr],yr_train)
        llw.fit(Xw_train[fw],yw_train)
        # metrics red
        pred_llr_tr = llr.predict(Xr_train[fr])
        rmse_trr,r2_tr = metrics_reg(yr_train,pred_llr_tr)
        pred_llr_v = llr.predict(Xr_val[fr])
        rmse_vr,r2_v = metrics_reg(yr_val,pred_llr_v)
        # metrics white
        pred_llw_tr = llw.predict(Xw_train[fw])
        rmse_trw,r2_tr = metrics_reg(yw_train,pred_llw_tr)
        pred_llw_v = llw.predict(Xw_val[fw])
        rmse_vw,r2_v = metrics_reg(yw_val,pred_llw_v)
        print('Lasso Lars        Red          White')
        print(f'Train       RMSE: {round(rmse_trr,2)}  RMSE:  {round(rmse_trw,2)}')
        print(f'Validate    RMSE: {round(rmse_vr,2)}  RMSE:  {round(rmse_vw,2)}')
    else:
        print('Please include model argument: lr, poly, tweedie, lasso')

def cluster_model(Xr_train,Xw_train,yr_train,yw_train,Xr_val,Xw_val,yr_val,yw_val):
    '''Input unscaled train and validate for cluster model result
    
    This will scale, cluster, add cluster to unscaled, then scale for modeling
    '''
    Xr_train_da,Xr_val_da = e.vol_sug_red_cluster(Xr_train,Xr_val)
    Xw_train_da,Xw_val_da = e.den_alc_white_cluster(Xw_train,Xw_val)
    # features
    fr=['total_so2_s_s', 'sulphates_s_s', 'alcohol_s_s', 'hi_acid_low_sug_s']
    fw=['fixed_acidity_s_s', 'volatile_acidity_s_s', 'free_so2_s_s', 'total_so2_s_s', 'sulphates_s_s', 'hi_den_low_alc_s', 'low_den_hi_alc_s', 'med_den_low_alc_s']
    # polynomial feature regression
    pfr = PolynomialFeatures(degree=3)
    pfw = PolynomialFeatures(degree=3)
    Xr_train_pf = pfr.fit_transform(Xr_train_da[fr])
    Xr_val_pf = pfr.transform(Xr_val_da[fr])
    Xw_train_pf = pfw.fit_transform(Xw_train_da[fw])
    Xw_val_pf = pfw.transform(Xw_val_da[fw])
    # model
    prr = LinearRegression()
    prw = LinearRegression()
    prr.fit(Xr_train_pf,yr_train)
    prw.fit(Xw_train_pf,yw_train)
    # metrics red
    pred_prr_tr = prr.predict(Xr_train_pf)
    rmse_trr,r2_tr = metrics_reg(yr_train,pred_prr_tr)
    pred_prr_v = prr.predict(Xr_val_pf)
    rmse_vr,r2_v = metrics_reg(yr_val,pred_prr_v)
    # metrics white
    pred_prw_tr = prw.predict(Xw_train_pf)
    rmse_trw,r2_tr = metrics_reg(yw_train,pred_prw_tr)
    pred_prw_v = prw.predict(Xw_val_pf)
    rmse_vw,r2_v = metrics_reg(yw_val,pred_prw_v)
    print('Linear Reg        Red          White')
    print(f'Train       RMSE: {round(rmse_trr,2)}  RMSE:  {round(rmse_trw,2)}')
    print(f'Validate    RMSE: {round(rmse_vr,2)}  RMSE:  {round(rmse_vw,2)}')

def test_model(Xr_train,Xw_train,yr_train,yw_train,Xr_test,Xw_test,yr_test,yw_test):
    '''Input scaled train and test data and it will return RMSE test results'''
    # features
    fr=['volatile_acidity_s', 'total_so2_s', 'sulphates_s', 'alcohol_s']
    fw=['fixed_acidity_s', 'volatile_acidity_s', 'citric_acid_s', 'chlorides_s', 'free_so2_s', 'total_so2_s', 'alcohol_s']
    # polynomial feature regression
    pfr = PolynomialFeatures(degree=3)
    pfw = PolynomialFeatures(degree=3)
    Xr_train_pf = pfr.fit_transform(Xr_train[fr])
    Xr_test_pf = pfr.transform(Xr_test[fr])
    Xw_train_pf = pfw.fit_transform(Xw_train[fw])
    Xw_test_pf = pfw.transform(Xw_test[fw])
    # model
    prr = LinearRegression()
    prw = LinearRegression()
    prr.fit(Xr_train_pf,yr_train)
    prw.fit(Xw_train_pf,yw_train)
    # metrics red
    pred_prr_t = prr.predict(Xr_test_pf)
    rmse_tr,r2_t = metrics_reg(yr_test,pred_prr_t)
    # metrics white
    pred_prw_t = prw.predict(Xw_test_pf)
    rmse_tw,r2_t = metrics_reg(yw_test,pred_prw_t)
    print('Poly Cluster        Red          White')
    print(f'Test          RMSE: {round(rmse_tr,2)}  RMSE:  {round(rmse_tw,2)}')

def plt_err(Xr_train,Xw_train,yr_train,yw_train,Xr_test,Xw_test,yr_test,yw_test):
    '''plot predicted vs actual property values by inputting train and test'''
    # features
    fr=['volatile_acidity_s', 'total_so2_s', 'sulphates_s', 'alcohol_s']
    fw=['fixed_acidity_s', 'volatile_acidity_s', 'citric_acid_s', 'chlorides_s', 'free_so2_s', 'total_so2_s', 'alcohol_s']
    # polynomial feature regression
    pfr = PolynomialFeatures(degree=3)
    pfw = PolynomialFeatures(degree=3)
    Xr_train_pf = pfr.fit_transform(Xr_train[fr])
    Xr_test_pf = pfr.transform(Xr_test[fr])
    Xw_train_pf = pfw.fit_transform(Xw_train[fw])
    Xw_test_pf = pfw.transform(Xw_test[fw])
    # model
    prr = LinearRegression()
    prw = LinearRegression()
    prr.fit(Xr_train_pf,yr_train)
    prw.fit(Xw_train_pf,yw_train)
    # metrics red
    pred_prr_t = prr.predict(Xr_test_pf)
    pred_mean_r = yr_test
    pred_mean_r = pred_mean_r.assign(baseline=pred_mean_r.quality.mean())
    # metrics white
    pred_prw_t = prw.predict(Xw_test_pf)
    pred_mean_w = yw_test
    pred_mean_w = pred_mean_w.assign(baseline=pred_mean_w.quality.mean())
    # plot stuff
    plt.figure(figsize=(16,5))
    # red subplot
    plt.subplot(121)
    plt.plot(yr_test, pred_mean_r.baseline, alpha=.5, color="red", label='_nolegend_')
    plt.annotate("Baseline: Mean", (3.2, 5.7))
    plt.plot(yr_test, yr_test, alpha=.5, color="red", label='_nolegend_')
    plt.annotate("Ideal: Predicted = Actual", (3.2, 3), rotation=34)
    plt.scatter(yr_test, pred_prr_t, alpha=.1, color="red", s=100, label="3rd degree Polynomial Red Predictions")
    plt.xlabel("Actual Wine Quality")
    plt.ylabel("Predicted Wine Quality")
    plt.title('Red Wine')
    # white subplot
    plt.subplot(122)
    plt.plot(yw_test, pred_mean_w.baseline, alpha=.5, color="grey", label='_nolegend_')
    plt.annotate("Baseline:   Mean", (3.2, 6.1))
    plt.plot(yw_test, yw_test, alpha=.5, color="grey", label='_nolegend_')
    plt.annotate("Ideal: Predicted = Actual", (2.9, 2.3), rotation=22)
    plt.scatter(yw_test, pred_prw_t, alpha=.1, color="grey", s=100, label="3rd degree Polynomial White Predictions")
    plt.xlabel("Actual Wine Quality")
    plt.ylabel("Predicted Wine Quality")
    plt.title('White Wine')
    plt.legend(loc='upper center')
    plt.suptitle("Where are predictions more extreme? More modest?")
    plt.show()