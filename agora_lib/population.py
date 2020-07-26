# import ray
import numpy as np
import pandas as pd
import random
from agora_lib import individual
import copy
from joblib import Parallel, delayed
import time
from google_utils.big_query import add_points_parallely
from config.globals import set_consumer_globals
#we'll use these global variables to share the memory for parallel processing
x_train_g=None
y_train_g=None
popSize_g=None
cutoff_g=4
from google_utils import big_query


class Population:
    popSize = 20;
    individuals = [None]*popSize
    cutoff=0;
    parents=[None]*cutoff;
    metric='neg_mean_squared_error'
    score_colors=[None]*popSize
    mut_rate=0.2


    # Initialize population
    def __init__(self,popSize, cutoff, x_train, y_train,cv, niter, attr): #no metric
        self.attr=attr
        self.popSize = popSize;
        self.attr=attr
        self.all_scores = np.zeros([0])
        # self.score_colors=[None]*popSize
        self.cutoff = cutoff;
        global popSize_g
        global x_train_g
        global y_train_g
        global cutoff_g
        x_train_g, y_train_g = x_train, y_train
        popSize_g, cutoff_g = self.popSize, self.cutoff
        self.cv=cv

        self.metric='neg_mean_squared_error'

        indvs=[individual.Individual(self.cv, self.metric) for _ in range(popSize)]
        self.individuals = indvs;
        self.parents=[None]*cutoff;
        for i in range(cutoff):
            self.parents[i]=indvs[i]
        self.niter=niter


    def get_pop_as_df(self):  # get population as a pandas data frame
        size = len(self.individuals)
        df = [None] * size;
        for i in range(size):
            df[i] = self.individuals[i].get_individual()
        return pd.DataFrame(df)

    def crossover(self):
        nSwap = 2;  # how many chromosomes to mutate

        inds = range(self.cutoff, self.popSize, 2)  # crossover every two strategies starting from the cut off point

        for i in inds:
            if hash(str(self.individuals[i].get_individual())) != hash(
                    str(self.individuals[i + 1].get_individual())):  # the individuals are different from each other

                cross_chroms = random.sample(['rem_bl', 'window', 'order', 'deriv', 'sc_method', 'ml_method'],
                                             nSwap)  # no ml to swap
                for cross_chrom in cross_chroms:
                    if cross_chrom == 'rem_bl':
                        self.individuals[i].rem_bl, self.individuals[i + 1].rem_bl = self.individuals[i + 1].rem_bl, \
                                                                                     self.individuals[i].rem_bl
                        self.individuals[i].lam, self.individuals[i + 1].lam = self.individuals[i + 1].lam, \
                                                                               self.individuals[i].lam
                        self.individuals[i].p, self.individuals[i + 1].p = self.individuals[i + 1].p, \
                                                                           self.individuals[i].p

                    elif cross_chrom == 'ml_method':  # swap ml method along with the parameters
                        for ml_detail in ['ml_method', 'ml_params']:
                            value = getattr(self.individuals[i], ml_detail)
                            setattr(self.individuals[i], ml_detail,
                                    getattr(self.individuals[i + 1], ml_detail))  # crossover
                            setattr(self.individuals[i + 1], ml_detail, value)  # crossover

                    else:  # swap savgol params or sc_method
                        value = getattr(self.individuals[i], cross_chrom)
                        setattr(self.individuals[i], cross_chrom,
                                getattr(self.individuals[i + 1], cross_chrom))  # crossover
                        setattr(self.individuals[i + 1], cross_chrom, value)  # crossover

    def cross_mut(self):
        self.crossover()
        # mutate
        for i in np.arange(cutoff_g, popSize_g, 1):
            if random.uniform(0, 1) < self.mut_rate:
                self.individuals[i].mutate()

    def gen_new_pop(self):  # generate a new population by parents
        for i in range(0, self.popSize, self.cutoff):
            for j in range(self.cutoff):
                self.individuals[i + j] = copy.deepcopy(self.parents[j])

    def evolve(self, scores, inds):
        idx = np.argsort(scores)[:cutoff_g]
        self.parents = [self.individuals[i] for i in inds[idx]]  # reassign individuals
        self.gen_new_pop()
        self.cross_mut()  # crossover-mutate population
        return scores[idx]  # best scores

    def eval_indv(self, idx):  # evaluate one individual
        return self.individuals[idx].calcFitness(x_train_g, y_train_g)

    def parallel_eval(self):
        df = self.get_pop_as_df()
        df.ml_params = df.ml_params.apply(lambda x: str(x))
        inds=df.iloc[cutoff_g:, :].drop_duplicates().index # evaluate only unique solutions 
        scores = np.array(Parallel(n_jobs=-1, require='sharedmem')(delayed(self.eval_indv)(i) for i in inds))
        return np.round(scores, 3), inds
    def push_ga_to_bq(self, x, y):

        add_points_parallely(metric="GA_scores", x_values=x, y_values=y, extras={"GA_iterations": 1})
        pass

    def iterate(self):
        t = time.time()
        scores = np.array(Parallel(n_jobs=-1, require='sharedmem')(
            delayed(self.eval_indv)(i) for i in np.arange(0, popSize_g, 1)))  # initial evaluation
        inds = np.arange(popSize_g)
        print('Initial evaluation time elapsed iteration %5.3f' % (time.time() - t))
        self.all_scores = np.append(self.all_scores, scores)  # append all scores)
        n_scores=len(scores)
        #self.push_ga_to_bq(x=np.arange(n_scores), y=scores)
        ### BIGQUERY Example ###

        x1=len(scores)
        score_inds=np.arange(x1)
        #self.push_ga_to_bq(x=score_inds, y=scores)

        for i in np.arange(1, self.niter + 1, 1):
            t = time.time()
            parent_scores = self.evolve(scores, inds)  # cross-over/mutate and reorder population based on scores
            child_scores, new_inds = self.parallel_eval()  # evaluate non-duplicated individuals
            scores = np.concatenate([parent_scores, child_scores])
            # self.push_ga_to_bq(x=np.arange(n_scores, n_scores+len(scores),1), y=scores)
            # n_scores+=len(scores)
            inds = np.concatenate([np.arange(cutoff_g), new_inds])


            print('Time elapsed for iteration ' + str(i) + ': %5.3f' % (time.time() - t))
            self.all_scores = np.append(self.all_scores, scores)  # append all scores)
        idx = scores.argsort()
        self.individuals = [self.individuals[i] for i in inds[idx]]  # reassign individuals
        k = 0
        for i in idx:  # assign scores since objects dont get changed during parallel computing
            self.individuals[k].fitness = scores[i]
            k += 1

    def evaluate(self):
        t = time.time()
        self.iterate()
        print('total time: %5.3f' % (time.time() - t))

