
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

import warnings
from sklearn.model_selection import validation_curve

warnings.filterwarnings("ignore")

import random
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats

import warnings
warnings.filterwarnings("ignore")


def rf(dspectra,Y, param, metric,cv=5):
    params = {'n_estimators': np.linspace(abs(param['n_estimators'] - 29), param['n_estimators'] + 50, 5, dtype=int),
              'min_samples_leaf': np.linspace(abs(param['min_samples_leaf'] - 0.04), param['min_samples_leaf'] + 0.05, 3),
              'min_samples_split': np.linspace(abs(param['min_samples_split'] - 0.04), param['min_samples_split'] + 0.05, 3)}

    gsc = RandomForestRegressor(n_jobs=-1, bootstrap=True, max_features=param['max_features'],
                                random_state=random.seed(1234))

    random_cv = RandomizedSearchCV(gsc, param_distributions=params, scoring=metric, cv=cv, n_jobs=-1, iid=False)
    random_cv.fit(dspectra, Y)
    if metric=='neg_mean_squared_error':
        score=np.sqrt(-random_cv.best_score_)
    else:
        score=random_cv.best_score_

    cv_param=random_cv.best_params_
    cv_param['max_features']=param['max_features']
    return score, cv_param

def rf_final_result(x_train, y_train,param):
    gsc = RandomForestRegressor(n_estimators=param['n_estimators'], bootstrap=True,
                                min_samples_split=param['min_samples_split'],
                                min_samples_leaf=param['min_samples_leaf'],
                                max_features=param['max_features'], n_jobs=-1,
                                random_state=random.seed(1234))

    gsc.fit(x_train, y_train);

    return gsc

def svr(dspectra,Y, param, metric,cv=5):

    params={'C': np.linspace(param['C']*.8,min(param['C']*1.2,400),3), 'gamma': np.linspace(param['gamma']*0.5, min(param['gamma']*1.5,0.008), 3)}

    svr = SVR(kernel='rbf')

    random_cv = RandomizedSearchCV(svr, param_distributions=params, scoring=metric, cv=cv, n_jobs=-1, iid=False)
    random_cv.fit(dspectra, Y)
    if metric == 'neg_mean_squared_error':
        score = np.sqrt(-random_cv.best_score_)
    else:
        score = random_cv.best_score_

    return score, random_cv.best_params_

def svr_final_result(x_train, y_train,  param):

    svr = SVR(kernel='rbf', C=round(param['C'],3),  gamma=round(param['gamma'],5))
    svr.fit(x_train, y_train);

    return svr

def pls(dspectra,Y,ncomp,metric,cv=5):
    # Run PLS including a variable number of components, up to 25,  and calculate RMSE
    maxComp=ncomp+3
    minComp=max(ncomp-3,0)


    param_range = np.arange(minComp, maxComp, 1)
    train_scores, test_scores = validation_curve(
        PLSRegression(), dspectra, Y, param_name="n_components", param_range=param_range,
        scoring=metric, cv=cv, n_jobs=-1)

    test_scores_mean = np.mean(test_scores, axis=1)
    ses = stats.sem(test_scores, axis=1)

    ind = test_scores_mean.argmax()
    score = test_scores_mean[ind]
    diff = abs(test_scores_mean - score)
    arg_param = np.argmax(diff < ses[ind])

    if metric=='neg_mean_squared_error':
        test_scores_mean=np.sqrt(-test_scores_mean)


    #     ind=scores.argmin()
    #     #Y=Y.flatten()
    #     # best_res=preds[ind,:]-Y
    #     #
    #     # if ind>0:
    #     #     for i in range(ind):
    #     #
    #     #         res=preds[i,:]-Y
    #     #         p=stats.f_oneway(best_res, res)[1]
    #     #         if p>0.05:
    #     #             ind=i
    #     #             break


    return [test_scores_mean[arg_param], param_range[arg_param]]

def pls_final_result(x_train, y_train,  param):


    pls=PLSRegression(param)

    pls.fit(x_train, y_train)
   # vips=prep.vip(x_train,pls )

    return pls






