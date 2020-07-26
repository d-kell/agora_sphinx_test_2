from agora_lib import prep
import sklearn as sk
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import numpy as np
import uuid


class Pipeline:
    def __init__(self,model_name,wl, rem_bl,lam, p, window, order, deriv, sc_method,ml_method, ml_params, fitness_cv,fitness_dev,fitness_dev_raw, ml_model):
    # als parameters
        self.model_name=model_name
        self.wl=wl
        self.rem_bl = rem_bl

        self.lam = lam
        self.p = p

        # Savitsky -Golay

        self.window = window
        self.order = order
        self.deriv = deriv

        # Scaling
        self.sc_method = sc_method
        # ml
        self.ml_method = ml_method
        self.ml_params = ml_params;
        self.ml_model=ml_model

    # Fitness
        self.fitness_cv = fitness_cv;
        self.fitness_dev=fitness_dev
        self.fitness_valid_raw=None
        self.fitness_dev_raw=fitness_dev_raw
        self.fitness_valid=None
        self.uuid = uuid.uuid1()
        self.r_line=None
        self.r2=None

    def prep_spectra(self, X_df):
        self.x_valid = X_df.loc[:, self.wl].to_numpy()


    def preprocess(self, fspec):

        if self.rem_bl:
            fspec = prep.remove_baseline(fspec, self.lam, self.p)
        fspec = savgol_filter(fspec, self.window, self.order, self.deriv)
        if self.sc_method == 0:
            return StandardScaler().fit_transform(fspec.T).T  # preprocessing.scale(fspec) #column- wise snv :0 mean 1 std
        elif self.sc_method == 1:
            fspec = sk.preprocessing.normalize(fspec, norm='max')
        # elif self.sc_method == 2:
        #     fspec = sk.preprocessing.normalize(fspec, norm='l1')
        # elif fspec=RobustScaler(quantile_range=(25, 75)).fit_transform(fspec)
        return fspec

    def predict(self, valid_spec):
        if 'simca' in self.model_name or 'SIMCA'  in self.model_name:
            self.y_pred = np.matmul(valid_spec, self.ml_model)
        else:
            self.y_pred= self.ml_model.predict(valid_spec)
        return self.y_pred

    def get_prediction(self, X_df):
        X_df = X_df.dropna(axis=0)
        self.prep_spectra(X_df)
        valid_spec=self.preprocess(self.x_valid)
        y_pred=self.predict(valid_spec)

        return y_pred