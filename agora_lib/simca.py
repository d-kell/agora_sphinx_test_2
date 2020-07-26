import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from rpy2.robjects.packages import importr
from sklearn.metrics import mean_squared_error, r2_score

simcaPls = importr('simcaNIPALS')
from agora_lib.base_model import Base_model
import warnings
from agora_lib import plotting, simca_pop

warnings.filterwarnings("ignore")
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
font_model_sec = 22


class SIMCA(Base_model):
    rmsecv = None
    y_pred = None
    cvIterator = None
    r_line = []
    rmsep = None
    rmsee = None

    def __init__(self, attr, val_set_given, plot_dir, wl, rep_ids,
                 rep_ids_val,  spec_dir, model_dir, accepted_err, niter,popSize, cutoff,**kwargs):

        Base_model.__init__(self, attr=attr, val_set_given=val_set_given, plot_dir=plot_dir, wl=wl, rep_ids=rep_ids,
                            rep_ids_val=rep_ids_val,  spec_dir=spec_dir, model_dir=model_dir,
                            accepted_err=accepted_err, niter=niter, popSize=popSize, cutoff=cutoff,**kwargs)



    def split_data(self, spectra, Y, x_valid, y_valid, pred_idx):

        self.cv=7

        self.x_train = spectra
        self.y_train = Y

        if self.val_set_given:
            self.x_valid = x_valid
            self.y_valid = y_valid
        else:
            self.x_valid = spectra
            self.y_valid = Y

        self.pred_idx=pred_idx

    def create_pop(self):
        self.pop = simca_pop.Population(popSize=self.popSize, cutoff=self.cutoff, x_train=self.x_train,
                                        y_train=self.y_train, cv=self.cv, niter=self.niter, attr=self.attr_name)  # create population
    def fit(self, ml_method='PLS'):

        result = simcaPls.plsNIPALS(self.train_x, self.y_train, ncomp=self.pipeline.ml_params + 1, it=50, tol=1e-6)
        ml_model = np.array(result[2][3]).reshape((self.x_train.shape[1], self.pipeline.ml_params))[:, -1]

        self.y_pred = np.array(np.matmul(self.train_x, ml_model))
        N=self.y_pred.shape[0]
        self.y_pred=self.y_pred.reshape([N,1])

        self.rmsee = round(np.sqrt(mean_squared_error(self.y_train, self.y_pred) * (N / (N - 1 - self.pipeline.ml_params))),3)

        self.pipeline.fitness_dev= self.rmsee

        self.pipeline.ml_model = ml_model

        result_raw = simcaPls.plsNIPALS(self.x_train, self.y_train, ncomp=self.pipeline.ml_params + 1, it=50, tol=1e-6)
        ml_model_raw = np.array(result_raw[2][3]).reshape((self.x_train.shape[1], self.pipeline.ml_params))[:, -1]

        self.y_pred_raw = np.array(np.matmul(self.x_train, ml_model_raw))
        N_raw = self.y_pred_raw.shape[0]
        self.y_pred_raw = self.y_pred_raw.reshape([N_raw, 1])

        self.pipeline.fitness_dev_raw = round(np.sqrt(mean_squared_error(self.y_train, self.y_pred_raw) * (N_raw / (N_raw - 1 - self.pipeline.ml_params))), 3)


    def save_figures(self):

        plotting.plot_spectra(wl=self.wl, spectra=self.valid_x, title=self.model_name + ' pipeline processed spectra:' + self.attr_name, save_fig=True, file_dir=self.pipeline.pr_spec_dir)
        # plotting.vip_plot(wl=self.wl, spectra=, coeffs, attr_name, file_dir)
        self.plot_corr(self.pipeline.corr_dir)



    def plot_corr(self, plot_dir):

        font = font_model_sec
        title = 'SIMCA pipeline correlation: ' + self.attr_name

        eps = np.finfo(np.float32).eps
        y_valid = self.y_valid
        y_pred = self.y_pred
        r2 = r2_score(y_valid, y_pred)
        z = np.polyfit(np.float32(y_valid.flatten()), y_pred, 1, rcond=len(y_valid) * eps)
        z = np.polyfit(np.float32(y_valid.flatten()), y_pred, 1, rcond=len(y_valid) * eps)
        self.pipeline.r_line = z

        rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': font, 'weight': 'light'})
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        ax.scatter(y_pred, y_valid, c='red', edgecolors='k')
        if z[1] > 0:
            label = "%5.2f $x$+ %5.2f"
        else:
            label = "%5.2f $x$ %5.2f"
        ax.plot(z[1] + z[0] * y_valid, y_valid, c='blue', linewidth=1, label=label % (z[0], z[1]))

        ax.plot(y_valid, y_valid, color='green', linewidth=1)

        plt.xlabel('Predicted')
        plt.ylabel('Measured')
   #     plt.title(title)
        xx = 0.18
        ax.legend(fontsize=font, loc="lower right")
        plt.figtext(xx, 0.75, 'R$^{2}$ : %5.3f' % r2, fontsize=font)
        plt.figtext(xx, 0.7, 'RMSE CV: %5.3f' % self.pipeline.fitness_cv, fontsize=font)
        y = 0.65
        if self.val_set_given:
            plt.figtext(xx, y, 'RMSEP : %5.3f' % self.pipeline.fitness_valid, fontsize=font)
        else:
            plt.figtext(xx, y, 'RMSEE : %5.3f' % self.pipeline.fitness_dev, fontsize=font)

        #         plt.figtext(xx, y-0.05, '95% CI:' + str(conf_interval(abs(y_pred.flatten()-y_valid.flatten()))), fontsize=font)

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
            # ax.get_xticklabels() + ax.get_yticklabels()
            item.set_fontsize(font)

        for item in ax.get_xticklabels():
            item.set_fontsize(font)
        for item in ax.get_yticklabels():
            item.set_fontsize(font)

        plt.tight_layout()

        plt.rcParams['font.size'] = font

        fig.savefig(plot_dir, bbox_inches="tight");
        plt.close()

