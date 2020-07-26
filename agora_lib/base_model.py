import warnings
from agora_lib import plotting, prep, population
from config.globals import set_consumer_globals
from google_utils.big_query import add_points_parallely

warnings.filterwarnings("ignore")
import pickle
from agora_lib.pipeline import Pipeline
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


class Base_model:
    pred_idx = None
    ml_model = None

    def __init__(self, model_name, attr, val_set_given, plot_dir, wl, rep_ids, rep_ids_val, pred_dir, pkl_dir, spec_dir,
                 model_dir, accepted_err=None, niter=10, popSize=20, cutoff=4, **kwargs):
        if model_name == 'simca':
            model_name = 'SIMCA'
        else:
            model_name = 'Agora'

        self.model_name = model_name
        self.val_set_given = val_set_given
        self.attr_name = attr
        self.rep_ids = rep_ids
        self.rep_ids_val = rep_ids_val
        self.wl = wl
        self.niter = niter
        self.popSize = popSize
        self.cutoff = cutoff
        # self.signal_col = signal_col
        # self.noise_col = noise_col
        self.name = attr.replace('/', '') + '_' + model_name
        self.pred_dir = pred_dir
        self.pkl_dir = pkl_dir
        self.spec_dir = spec_dir
        self.model_dir = model_dir
        self.accepted_err = accepted_err
        self.ga_scores = prep.make_subf(plot_dir, 'GA_scores')
        self.pop_dir = prep.make_subf(plot_dir, 'Last_Generation_Population')
        self.vip_dir = prep.make_subf(plot_dir, 'Variable Importance')
        self.stop_flag = False

        # get a list of all predefined values directly from __dict__
        allowed_keys = list(self.__dict__.keys())

        # Update __dict__ but only for keys that have been predefined
        # (silently ignore others)
        self.__dict__.update((key, value) for key, value in kwargs.items()
                             if key in allowed_keys)

        # To NOT silently ignore rejected keys
        rejected_keys = set(kwargs.keys()) - set(allowed_keys)
        if rejected_keys:
            raise ValueError("Invalid arguments in constructor:{}".format(rejected_keys))

        self.__dict__.update(kwargs)

    def split_data(self, spectra, Y, x_valid, y_valid, pred_idx):
        pass
    def create_pop(self):
        pass

    def train(self):
        self.pop = population.Population(popSize=self.popSize, cutoff=self.cutoff, x_train=self.x_train,
                                         y_train=self.y_train, cv=self.cv,niter=self.niter, attr=self.attr_name)  # create population

        self.create_pop()
        self.pop.evaluate()  #### GA RESULTS

        # save model
        best_indv = self.pop.individuals[0]  # save the best(fittest) pipeline

        self.pipeline = Pipeline(model_name=self.model_name, wl=self.wl, rem_bl=best_indv.rem_bl, lam=best_indv.lam,
                                 p=best_indv.p, window=best_indv.window,
                                 order=best_indv.order,
                                 deriv=best_indv.deriv, sc_method=best_indv.sc_method, ml_method=best_indv.ml_method,
                                 ml_params=best_indv.ml_params, fitness_cv=round(best_indv.fitness,3), fitness_dev=None,
                                 fitness_dev_raw= None, ml_model=None)
        self.pipeline.ga_scores_dir = "{}/{}".format(self.ga_scores,
                                            self.name + '_ga_scores.jpg')  # create a sub folder to output the results
        self.pipeline.pr_spec_dir = "{}/{}".format(self.spec_dir, self.name + '.jpg')
        self.pipeline.raw_attr_spec_dir = "{}/{}".format(self.spec_dir, self.name + '_raw.jpg')
        self.pipeline.vip_bar_dir = "{}/{}".format(self.vip_dir, self.name + '_vip.jpg')
        self.pipeline.vip_spec_dir = "{}/{}".format(self.vip_dir, self.name + '_spec_vip.jpg')
        self.pipeline.infl_dir = "{}/{}".format(self.model_dir, self.name + '_influence.jpg')  # influence plot
        self.pipeline.pred_err_dir = "{}/{}".format(self.model_dir, self.name + 'pred_err.jpg')  # pred error
        self.pipeline.corr_dir = "{}/{}".format(self.model_dir, self.name + '_corr.jpg')
        plotting.ga_scores(scores=self.pop.all_scores, title='Fitness CV Evolution: ' + self.name,
                           file_dir=self.pipeline.ga_scores_dir)
        self.pop.get_pop_as_df().to_csv(
            self.pop_dir + "/" + self.name + "_population.csv")  # save final population as csv

    def preprocess(self):

        self.train_x = self.pipeline.preprocess(self.x_train)

        if self.model_name == 'SIMCA' or self.model_name=='simca':
            self.valid_x = self.pipeline.preprocess(self.x_valid)
            plotting.plot_spectra(wl=self.wl, spectra=self.x_valid, title='Raw Spectra: ' + self.attr_name,
                                  save_fig=True,
                                  file_dir=self.pipeline.raw_attr_spec_dir)
        else:  # agora pipeline
            self.dev_x = self.pipeline.preprocess(self.x_dev)
            plotting.plot_spectra(wl=self.wl, spectra=self.x_dev, title='Raw Test Spectra: ' + self.attr_name,
                                  save_fig=True,
                                  file_dir=self.pipeline.raw_attr_spec_dir)
            if self.val_set_given:
                self.valid_x = self.pipeline.preprocess(self.x_valid)
                plotting.plot_spectra(wl=self.wl, spectra=self.x_valid, title='Raw Validation Spectra: ' + self.attr_name,
                                      save_fig=True,
                                      file_dir=self.pipeline.raw_attr_spec_dir)

    # def get_snr(self):
    #
    #     for i in range(self.x_train.shape[1]):
    #         if np.isin(self.x_train[:, i], self.noise_col).all() == True:
    #             self.noise_idx = i
    #         if np.isin(self.x_train[:,i], self.signal_col).all() == True:
    #             self.signal_idx = i
    #     signal_raw = self.x_train[:, self.signal_idx]
    #     noise_raw = self.x_train[:, self.noise_idx]
    #     max_signal_val_raw = np.max(self.x_train)
    #     max_signal_val = np.max(self.train_x)
    #     SNR_raw = (((signal_raw - noise_raw) / np.sqrt(abs(noise_raw))).mean()) / max_signal_val_raw
    #     signal = self.train_x[:, self.signal_idx]
    #     noise = self.train_x[:, self.noise_idx]
    #     SNR = (((signal - noise) / np.sqrt(abs(noise))).mean()) / max_signal_val
    #
    #     return SNR, SNR_raw


    def fit(self, ml_method):
        pass

    def predict(self):

        if self.val_set_given:
            self.y_pred = self.pipeline.predict(self.valid_x)
            N = self.y_pred.shape[0]
            self.y_pred = self.y_pred.reshape([N])
            self.pipeline.r2=r2_score(self.y_pred, self.y_valid)
            self.pipeline.fitness_valid = round(np.sqrt(mean_squared_error(self.y_valid, self.y_pred)), 3)
            self.y_pred_raw = self.pipeline.predict(self.x_valid)
            N_raw = self.y_pred_raw.shape[0]
            self.y_pred_raw = self.y_pred_raw.reshape([N_raw])
            self.pipeline.fitness_valid_raw = round(np.sqrt(mean_squared_error(self.y_valid, self.y_pred_raw)), 3)


    def save_figures(self):
        pass

    def save_predictions(self):

        if not self.val_set_given and self.model_name=='Agora':
            y_true=self.y_dev.flatten()
        else:
            y_true=self.y_valid.flatten()
        self.y_pred=self.y_pred.flatten()

        diff = self.y_pred - y_true
        rel_error = np.round(diff / y_true, 3).flatten()
        data = np.array([y_true, self.y_pred, diff, rel_error]).T
        df = pd.DataFrame(data=data, columns=[self.attr_name + ' Meas.', self.name,
                                              self.attr_name + ' Absolute Error', self.attr_name + ' Relative Error'],
                          index=self.pred_idx)
        df.reset_index(inplace=True)
        df.to_csv(self.pred_dir + '/Predictions_' + self.name + '.csv')

        plotting.infl_plot(df.iloc[:, 1], df.iloc[:, 2], attr_name=self.attr_name, plot_dir=self.pipeline.infl_dir)

        ### BIGQUERY Example ###
        #set_consumer_globals('random', self.model_name, self.attr_name)

        add_points_parallely(metric="correlation", x_values=y_true, y_values=self.y_pred, extras={"correlation": 1})

        plotting.pred_err_plot(y_true, self.y_pred, file_dir=self.pipeline.pred_err_dir,
                               title=self.model_name + ' Pipeline Prediction Error', attr_name=self.attr_name,
                               accepted_err=self.accepted_err)

    def save_model(self):

        f = open(self.pkl_dir + '/' + self.name + '.pkl', 'wb')
        pickle.dump(self.pipeline, f)

    def get_results(self):
        self.train()
        self.preprocess()
        # self.pipeline.snr, self.pipeline.snr_raw = self.get_snr()
        self.fit(ml_method=self.pipeline.ml_method)
        self.predict()
        self.save_model()
        self.save_figures()
        self.save_predictions()
        return self.pipeline
