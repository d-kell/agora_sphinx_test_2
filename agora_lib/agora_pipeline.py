import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from agora_lib import plotting, ml, prep, population

warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupShuffleSplit
from sklearn.cross_decomposition import PLSRegression
import random
from agora_lib.base_model import Base_model
from sklearn.model_selection import validation_curve


class Agora_pipeline(Base_model):
    train_x = []
    dev_x = []
    y_pred = []
    r_line = []
    rmsecv = None
    rmsep = None
    cv=5


    def __init__(self, attr, val_set_given, plot_dir, wl, rep_ids,
                 rep_ids_val,  spec_dir, model_dir, accepted_err,niter,popSize, cutoff, **kwargs):

        Base_model.__init__(self, attr=attr, val_set_given=val_set_given, plot_dir=plot_dir, wl=wl, rep_ids=rep_ids,
                            rep_ids_val=rep_ids_val,  spec_dir=spec_dir, model_dir=model_dir,
                            accepted_err=accepted_err, niter=niter, popSize=popSize, cutoff=cutoff,**kwargs)


    def split_data(self, spectra, Y, x_valid, y_valid, pred_idx):

        if spectra.shape[0] > 500:
            self.cv=10
        else:
            self.cv=5

        test_size=round(1/(1+self.cv),2) # leave one fold out for model characterization


        if len(self.rep_ids) > 0:  # replicates are present

            if self.val_set_given:
                train_inds, test_inds = next(
                    GroupShuffleSplit(test_size=test_size, n_splits=2, random_state=7).split(self.rep_ids,
                                                                                             groups=self.rep_ids))
                x_train, x_test, y_train, y_test = spectra[train_inds, :], spectra[test_inds, :], Y[train_inds], Y[
                    test_inds]
                # x_train, x_test, y_train, y_test = train_test_split(spectra, Y, test_size=test_size, random_state=1)
                self.pred_idx = pred_idx
                self.x_valid, self.y_valid = x_valid, y_valid

            else:
                train_inds, test_inds = next(
                    GroupShuffleSplit(test_size=test_size, n_splits=2, random_state=7).split(self.rep_ids,
                                                                                             groups=self.rep_ids))
                x_train, x_test, y_train, y_test = spectra[train_inds, :], spectra[test_inds, :], Y[train_inds], Y[
                    test_inds]
                self.pred_idx = test_inds
        else:  # no replicates
            if self.val_set_given:
                self.pred_idx = pred_idx
                x_train, x_test, y_train, y_test = train_test_split(spectra, Y, test_size=test_size, random_state=1)

                self.x_valid, self.y_valid = x_valid, y_valid

            else:
                x_train, x_test, y_train, y_test, train_inds, test_inds = train_test_split(spectra, Y, pred_idx,
                                                                                           test_size=test_size,
                                                                                           random_state=random.seed(
                                                                                               1234))
                self.pred_idx = test_inds
        self.x_dev = x_test
        self.y_dev = y_test
        self.x_train = x_train
        self.y_train = y_train


    def create_pop(self):
        self.pop = population.Population(popSize=self.popSize, cutoff=self.cutoff, x_train=self.x_train,
                                        y_train=self.y_train, cv=self.cv, niter=self.niter, attr=self.attr_name)  # create population

    def fit(self, ml_method):
        train_scores, test_scores, param_range, param_name=[],[],[],''
        import math
        import scipy.stats as st

        if ml_method == 'SVR':

            ml_model = ml.svr_final_result(self.train_x, self.y_train,
                                           self.pipeline.ml_params)
            ml_model_raw = ml.svr_final_result(self.x_train, self.y_train,
                                               param=self.pipeline.ml_params)
            param_name="gamma"
            # c_power = math.log2(self.pipeline.ml_params[param_name])
            # c_min, c_max = max(c_power - 2, -1), min(c_power + 2, 8)
            # c_num =int( c_max - c_min + 1)
            # param_range=np.logspace(c_min, c_max, base=2,num=c_num)
            g_power = math.log2(self.pipeline.ml_params[param_name])
            g_min, g_max = g_power - 3, min(g_power + 2, -2)
            g_num = int(g_max - g_min + 1)
            param_range=np.logspace(g_min, g_max, base=2,num=g_num)


            train_scores, test_scores = validation_curve(
                ml_model, self.train_x, self.y_train, param_name=param_name, param_range=param_range,
                scoring=self.pop.metric, cv=self.cv, n_jobs=-1)
            test_scores_mean = np.mean(test_scores, axis=1)

            ses = st.sem(test_scores, axis=1)
            ind = test_scores_mean.argmax()
            score = test_scores_mean[ind]
            diff = abs(test_scores_mean - score)
            arg_param = np.argmax(diff < ses[ind])

            if param_range[arg_param] !=self.pipeline.ml_params[param_name]:
                self.pipeline.ml_params[param_name]=param_range[arg_param]
                ml_model = ml.svr_final_result(self.train_x, self.y_train,
                                           param=self.pipeline.ml_params)
                ml_model_raw= ml.svr_final_result(self.x_train, self.y_train,
                                           param=self.pipeline.ml_params)



        elif ml_method == 'RF':
            ml_model = ml.rf_final_result(self.train_x, self.y_train,
                                          self.pipeline.ml_params)


        elif ml_method == 'PLS':


            ml_model = ml.pls_final_result(self.train_x, self.y_train,
                                           param=self.pipeline.ml_params)
            ml_model_raw = ml.pls_final_result(self.x_train, self.y_train,
                                               param=self.pipeline.ml_params)

            maxComp = min(self.pipeline.ml_params + 5, 15)
            minComp = max(self.pipeline.ml_params - 3, 1)

            param_range = np.arange(minComp, maxComp, 1)
            param_name="n_components"
            train_scores, test_scores = validation_curve(
                ml_model, self.train_x, self.y_train, param_name=param_name, param_range=param_range,
                scoring=self.pop.metric, cv=self.cv, n_jobs=-1)

            test_scores_mean = np.mean(test_scores, axis=1)

            ses = st.sem(test_scores, axis=1)
            ind = test_scores_mean.argmax()
            score = test_scores_mean[ind]
            diff = abs(test_scores_mean - score)
            arg_param = np.argmax(diff < ses[ind])

            if param_range[arg_param] !=self.pipeline.ml_params:
                self.pipeline.ml_params=param_range[arg_param]
                ml_model = ml.pls_final_result(self.train_x, self.y_train,
                                           param=self.pipeline.ml_params)
                ml_model_raw=ml.pls_final_result(self.x_train, self.y_train,
                                           param=self.pipeline.ml_params)



        self.y_pred = ml_model.predict(self.dev_x)  # test/dev prediction
        self.pipeline.fitness_dev= round(np.sqrt(mean_squared_error(self.y_dev, self.y_pred)), 3)
        # train and predict on raw
        self.y_pred_raw = ml_model_raw.predict(self.x_dev)
        self.pipeline.fitness_dev_raw = round(np.sqrt(mean_squared_error(self.y_dev, self.y_pred_raw)), 3)

        self.pipeline.ml_model = ml_model
        self.pipeline.ml_selection_dir=self.model_dir+'/'+self.name+'_model_selection.jpg'
        plotting.plot_val_curve(train_scores=train_scores,test_scores=test_scores,param_range=param_range, param_name=param_name, metric=self.pop.metric, ml_method=self.pipeline.ml_method, fig_dir=self.pipeline.ml_selection_dir)


    def save_figures(self):


        if self.val_set_given:
            title = 'Universal pipeline spectra: ' + self.attr_name

            plotting.plot_spectra(self.wl, self.valid_x, title=title, save_fig=True, file_dir=self.pipeline.pr_spec_dir)

            if isinstance(self.pipeline.ml_model, PLSRegression):
                coeffs = prep.vip(self.x_valid, self.pipeline.ml_model)
                x=coeffs[~np.isnan(coeffs)]
                if len(x)>0:
                    self.pipeline.vips = coeffs


                    idx = np.argsort(coeffs)[::-1]
                    pd.DataFrame(data=coeffs[idx], index=self.wl[idx], columns=['Variable importance coefficient']).to_csv(
                        self.vip_dir + '/_vip-'+self.name+'.csv')

                    plotting.vip_plot2(self.wl, self.x_valid, coeffs, self.name, file_dir=self.pipeline.vip_spec_dir)
                    plotting.vip_bar(self.wl, coeffs, self.attr_name, file_dir=self.pipeline.vip_bar_dir)

        else:
            title = 'Universal pipeline processed spectra: ' + self.attr_name
            plotting.plot_spectra(self.wl, self.dev_x, title=title, save_fig=True, file_dir=self.pipeline.pr_spec_dir)
            if isinstance(self.pipeline.ml_model, PLSRegression):
                coeffs = prep.vip(self.x_train, self.pipeline.ml_model)
                x = coeffs[~np.isnan(coeffs)]
                if len(x) > 0:
                    self.pipeline.vips = coeffs

                    idx = np.argsort(coeffs)[::-1]
                    pd.DataFrame(data=coeffs[idx], index=self.wl[idx],
                                 columns=['Variable importance coefficient']).to_csv(
                        self.vip_dir + '/_vip'+self.name+'.csv')

                    plotting.vip_plot2(self.wl, self.x_dev, coeffs, self.name, file_dir=self.pipeline.vip_spec_dir)
                    plotting.vip_bar(self.wl, coeffs, self.attr_name, file_dir=self.pipeline.vip_bar_dir)


        if self.val_set_given:

            self.pipeline.r_line = plotting.corr_plot_univ(y_pred=self.y_pred, y_test=self.y_valid, fitness_cv=self.pipeline.fitness_cv,
                                                           fitness_test=self.pipeline.fitness_dev,
                                                           title='Universal pipeline correlation: ' + self.attr_name,
                                                           file_dir=self.pipeline.corr_dir,
                                                           val_set_given=self.val_set_given, fitness_valid=self.pipeline.fitness_valid)
        else:
            self.pipeline.r_line = plotting.corr_plot_univ(y_pred=self.y_pred, y_test=self.y_dev, fitness_cv=self.pipeline.fitness_cv,
                                                           fitness_test=self.pipeline.fitness_dev,
                                                           title='Universal pipeline correlation: ' + self.attr_name,
                                                           file_dir=self.pipeline.corr_dir,
                                                           val_set_given=self.val_set_given)


