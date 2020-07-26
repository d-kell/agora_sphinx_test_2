
import pandas as pd
from agora_lib import ml, prep
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import sklearn as sk
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import validation_curve
import random
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats

import warnings
warnings.filterwarnings("ignore")


# num_cpus=psutil.cpu_count(False)
# ray.init(num_cpus=psutil.cpu_count(logical=False),ignore_reinit_error=True) #initialize ray
from sklearn.preprocessing import RobustScaler

# @ray.remote
class Individual:
    fitness = 0;

    # ml method
    ml_method = 'SVR'
    # hyperparameter ranges
    svr_C = list(np.logspace(-1, 9, base=2, num=11))

    svr_gamma = list(np.logspace(-8, -2, base=2, num=7))
    pls_range=np.arange(1,15,1)

    nMut=1
    lam_range=[4,5,6,7,8]
    p_range=[3,5,7,9]

    w_range=[5,7,9,11,13,15,17]
    or_range=[2,3,4]
    d_range=[0,1,2]
    sc_range=[0,1,2,3]


    ml_params = {'C': random.choice(svr_C), 'gamma': random.choice(svr_gamma)}


    def __init__(self, cv, metric ):
        # self.rf_n_estimators = list(np.arange(70, 200, 30))
        # self.rf_max_features = ['auto', 'sqrt', 'log2']
        # self.rf_min_samples_leaf = list(np.linspace(0.1, 0.4, 4, endpoint=True))
        # self.rf_min_samples_split = list(np.linspace(0.2, 0.8, 7, endpoint=True))
        self.cv=cv
        self.metric=metric

        # als parameters
        self.rem_bl = random.randint(0, 1);
        if self.rem_bl:
            self.lam = random.choice(self.lam_range)
            self.p = random.choice(self.p_range)
        else:
            self.lam=0
            self.p=0

        # Savitsky -Golay

        self.window = random.choice(self.w_range)
        self.order = random.choice(self.or_range)
        self.deriv = random.choice(self.d_range)

        # Scaling
        self.sc_method = random.choice(self.sc_range);
        # ml
        self.ml_method = random.choice(['PLS',  'SVR']);
        if self.ml_method == 'PLS':
            self.ml_params = 2;
        else:
            # if self.ml_method == 'SVR':
            self.ml_params = {'C': random.choice(self.svr_C),
                                  'gamma': random.choice(self.svr_gamma)}

        self.fitness = -1;

    def __eq__(self, other):
        if not isinstance(other, Individual):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return \
                 self.ml_method == other.ml_method and \
                 self.ml_params ==other.ml_params and \
                 self.window == other.window and \
                 self.order == other.order and \
                 self.deriv==other.deriv and \
                 self.rem_bl==other.rem_bl and \
                 self.lam==other.lam and \
                 self.p==other.p  and \
                 self.sc_method==other.sc_method


    def get_fitness(self):

        return self.fitness

    def get_individual(self):

        if self.rem_bl :
            res={'bl_method':'ALS', 'als_l':self.lam, 'als_p':self.p, 'sg_window': self.window, 'sg_order':self.order,'sg_deriv':self.deriv,'sc_method': self.sc_method, 'ml_method': self.ml_method,
        'ml_params': self.ml_params, 'fitness CV': self.fitness}
        else:
            res={'bl_method':'None', 'als_l': 'NA', 'als_p': 'NA','sg_window': self.window, 'sg_order':self.order,'sg_deriv':self.deriv,'sc_method': self.sc_method, 'ml_method': self.ml_method,
        'ml_params': self.ml_params, 'fitness CV': self.fitness}
        return res

    def set_sim_indv(self,sim_obj):

        self.rem_bl=0
        self.lam=0
        self.p=0

        self.window=sim_obj.window
        self.order=sim_obj.order
        self.deriv=sim_obj.deriv

        self.sc_method=0

        self.ml_method='PLS'
        self.ml_params=sim_obj.comp
      #  self.fitness=sim_obj.rmsecv


    def preprocess(self, fspec):

        if self.rem_bl:
            fspec = prep.remove_baseline(fspec, self.lam, self.p)
        fspec = savgol_filter(fspec, self.window, self.order, self.deriv)
        if self.sc_method == 0:
            return StandardScaler().fit_transform(fspec.T).T  # preprocessing.scale(fspec) #column- wise snv :0 mean 1 std
            # fspec=RobustScaler(quantile_range=(25, 75)).fit_transform(fspec) #
        elif self.sc_method == 1:
            fspec = sk.preprocessing.normalize(fspec, norm='max')
        # elif self.sc_method== 4:
        #    fspec = sk.preprocessing.normalize(fspec, norm='l1')
            #fspec=RobustScaler(quantile_range=(25, 75)).fit_transform(fspec.T).T
        return fspec

    def calcFitness(self, spectra,Y):

        dspectra = self.preprocess(spectra)
        ml_method = self.ml_method

        if ml_method == 'SVR':

            score, param = self.svr(dspectra, Y, self.ml_params, self.metric, cv=self.cv)
            self.fitness = score

            self.ml_params = param

        elif ml_method == 'PLS':

            [score, param] = self.pls(dspectra, Y, self.ml_params, metric=self.metric, cv=self.cv)
            self.fitness = score
            self.ml_params =param


        return score

    def mut_window(self):
        windws=self.w_range.copy()
        windws.remove(self.window)

        self.window =random.choice(windws) #mutate current window size to a different window size

    def mut_order(self):

        ords = self.or_range.copy()
        ords.remove(self.order)
        self.order= random.choice(ords)  # mutate current window size to a different window size

    def mut_deriv(self):

        drvs=self.d_range.copy()
        drvs.remove(self.deriv)
        self.deriv=random.choice(drvs)

    def mut_sc(self):

        scs=self.sc_range.copy()
        scs.remove(self.sc_method)

        self.sc_method=random.choice(scs)

    def mut_als_l(self):

        lams=self.lam_range.copy()
        lams.remove(self.lam)
        self.lam=random.choice(lams)

    def mut_als_p(self):
        alsp = self.p_range.copy()
        alsp.remove(self.p)
        self.p = random.choice(alsp)
    def mut_svr(self,key):
        if key=='C':
            params = self.svr_C.copy()
        else:
            params = self.svr_gamma.copy()
        # print(key)
        # print(params)
        # print(self.ml_params.keys())
        value=self.ml_params[key]
        # print(value)
        if value in params:
             params.remove(value)
        self.ml_params[key]=random.choice(params)

    def mut_ml_mtd(self):
        #only pls or svr
        if self.ml_method == 'PLS':
            self.ml_method='SVR'
            self.ml_params = {'C': random.choice(self.svr_C),
                              'gamma': random.choice(self.svr_gamma)}

        else:
            self.ml_method='PLS'
            self.ml_params = 5;



    def mutate(self):

        if self.ml_method=='PLS':
            ml_genes=0
        else:
            ml_genes=len(self.ml_params)

       # nMut= round(0.2 * chsLen)  # ttl number of genes to mutate per chromosome

        # for all genes to have same probability, add weights per subset (=to number of genes per subset)

        genes = ['window'] * 1+['order']*1 +['deriv']*1+ ['sc_method'] * 1 + ['ml_method']+['ml_params'] * ml_genes
        if self.rem_bl:
            genes += ['lam']+['p']
        else:
            genes += ['rem_bl']

        # sample nutating subsets
        genes2mut= pd.Series(random.sample(genes, self.nMut))

        ml_genes=sum(genes2mut=='ml_params')
        #nMut at most 2!
        if ml_genes==2: # mutate both svr params
           self.mut_svr('C')
           self.mut_svr('gamma')

        else:
            for gene in genes2mut:
                if gene=='rem_bl':
                    if self.rem_bl:
                        self.lam = 0
                        self.p = 0
                        self.rem_bl=0


                    else:
                        self.lam = random.choice(self.lam_range)
                        self.p = random.choice(self.p_range)
                        self.rem_bl=1

                elif gene=='lam':
                    self.mut_als_l()

                elif gene=='p':
                    self.mut_als_p()

                elif gene=='window':
                    self.mut_window()

                elif gene=='order':
                    self.mut_order()

                elif gene=='deriv':
                    self.mut_deriv()

                elif gene=='sc_method':
                    self.mut_sc()
                elif gene=='ml_method':
                    self.mut_ml_mtd()

                elif gene=='ml_params':
                    param=random.choice(['C','gamma'])
                    self.mut_svr(param)


    def rf(self,dspectra,Y, param, metric,cv=5):
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

    def rf_final_result(self,x_train, y_train,param):
        gsc = RandomForestRegressor(n_estimators=param['n_estimators'], bootstrap=True,
                                    min_samples_split=param['min_samples_split'],
                                    min_samples_leaf=param['min_samples_leaf'],
                                    max_features=param['max_features'], n_jobs=-1,
                                    random_state=random.seed(1234))

        gsc.fit(x_train, y_train);

        return gsc

    def svr(self,dspectra,Y, param, metric,cv=5):
        import math
        c_power , g_power = math.log2(param['C']) ,math.log2(param['gamma'])

        g_min, g_max  = max(g_power - 3, -14), min(g_power + 3, -2)
        c_min, c_max  = max(c_power - 2,-1), min(c_power + 3, 9)

        c_num=c_max-c_min +1
        g_num = g_max - g_min + 1

        params = {'C': np.logspace(c_min, c_max, base=2, num= int(c_num)),
                  'gamma': np.logspace(g_min, g_max, base=2, num= int(g_num))}

        svr = SVR(kernel='rbf')

        random_cv = RandomizedSearchCV(svr, param_distributions=params, scoring=metric, cv=cv, n_jobs=-1, iid=False)
        random_cv.fit(dspectra, Y)
        if metric == 'neg_mean_squared_error':
            score = np.sqrt(-random_cv.best_score_)
        else:
            score = random_cv.best_score_

        return score, random_cv.best_params_

    def svr_final_result(self,x_train, y_train,  param):

        svr = SVR(kernel='rbf', C=round(param['C'],3),  gamma=round(param['gamma'],5))
        svr.fit(x_train, y_train);

        return svr

    def pls(self,dspectra,Y,ncomp,metric,cv=5):
        # Run PLS including a variable number of components, up to 25,  and calculate RMSE
        maxComp=ncomp+5
        minComp=max(ncomp-3,1)


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

    def pls_final_result(self,x_train, y_train,  param):


        pls=PLSRegression(param)

        pls.fit(x_train, y_train)
       # vips=prep.vip(x_train,pls )

        return pls






