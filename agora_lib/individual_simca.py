import random
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import numpy as np
import pathos.pools as pp
from rpy2.robjects.packages import importr
from scipy import stats
from sklearn.metrics import mean_squared_error

simcaPls = importr('simcaNIPALS')
import warnings
import pandas as pd
from agora_lib import prep
import psutil
warnings.filterwarnings("ignore")
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

class SIMCA_Individual:
    fitness = 0;

    nMut=1
    w_range=[5,7,9,11,13,15,17]
    or_range=[2,3,4]
    d_range=[0,1]
    sc_range=[0,1]

    def __init__(self, cv, metric):

        self.cv=cv
        self.metric=metric
        self.window = random.choice(self.w_range)
        self.order = random.choice(self.or_range)
        self.deriv = random.choice(self.d_range)
        # Scaling
        self.sc_method = random.choice(self.sc_range);
        self.ml_method = 'PLS';
        self.ml_params = 2;
        self.fitness = -1;

    def get_fitness(self):

        return self.fitness
    def get_individual(self):
        res={'bl_method':'None', 'als_l': 'NA', 'als_p': 'NA','sg_window': self.window, 'sg_order':self.order,'sg_deriv':self.deriv,'sc_method': self.sc_method, 'ml_method': self.ml_method,
        'ml_params': self.ml_params, 'fitness CV': self.fitness}
        return res

    def preprocess(self, fspec):
        fspec = savgol_filter(fspec, self.window, self.order, self.deriv)
        if self.sc_method == 0:
            return StandardScaler().fit_transform(fspec.T).T  # preprocessing.scale(fspec) #column- wise snv :0 mean 1 std
        return fspec
    def get_cv_inds(self, n):
        labels = np.array([2, 3, 4, 5, 6])
        labels = np.concatenate([labels, np.tile(np.arange(7), int((n - 6) / 7))])
        for i in range(n - labels.shape[0]):
            labels = np.append(labels, i)
        cvIterator = []
        nFolds = 7
        for i in range(nFolds):
            trainIndices = np.where(labels != i)[0]
            testIndices = np.where(labels == i)[0]
            cvIterator.append((trainIndices, testIndices))
        return cvIterator
    def cv_test(self, params):
        train_index = params[0]
        test_index = params[1]
        m = self.dspectra.shape[1]

        x_train, x_test = self.dspectra[train_index], self.dspectra[test_index]
        y_train, y_test = self.y_train[train_index], self.y_train[test_index]
        result = simcaPls.plsNIPALS(x_train, y_train, ncomp=self.ncomp + 1, it=50, tol=1e-6)
        B = np.array(result[2][3]).reshape((m, self.ncomp))[:, -1]
        y_pred = np.matmul(x_test, B)
        return np.sqrt(mean_squared_error(y_pred, y_test))


    def cross_valid(self):
        self.cvIterator = self.get_cv_inds(self.dspectra.shape[0])
        p = pp.ProcessPool(processes=psutil.cpu_count(True) - 1)  # num cpus = n_available-1
        return np.array(p.map(self.cv_test, self.cvIterator))

    def calcFitness(self):

        maxComp = self.ml_params+4+1 #include edges
        minComp = max(self.ml_params-2, 1)
        rmses = np.zeros([maxComp - minComp])
        ses = np.zeros([maxComp - minComp])
        self.dspectra = prep.simca_prep(self.x_train, self.window, self.order, self.deriv)

        for ncomp in np.arange(minComp, maxComp, 1):
            self.ncomp=ncomp
            errs = self.cross_valid()
            rmses[ncomp - minComp] = errs.mean()
            ses[ncomp - minComp] = stats.sem(errs)
        ind = rmses.argmin()
        score = rmses[ind]
        diff = abs(rmses - score)
        comp = np.argmax(diff < ses[ind])
        score = rmses[comp]
        comp += minComp
        self.ml_params=comp

        return comp, score

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


    def mutate(self):
        genes = ['window'] * 1+['order']*1 +['deriv']*1+ ['sc_method'] * 1  # for all genes to have same probability, add weights per subset (=to number of genes per subset)

        genes2mut= pd.Series(random.sample(genes, self.nMut))

        for gene in genes2mut:

            if gene=='window':
                self.mut_window()

            elif gene=='order':
                self.mut_order()

            elif gene=='deriv':
                self.mut_deriv()

            elif gene=='sc_method':
                self.mut_sc()
