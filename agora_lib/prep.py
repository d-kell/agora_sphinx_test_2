import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

from scipy import sparse
from scipy.sparse.linalg import spsolve
from agora_lib import plotting
import rpy2.robjects as robjects
import multiprocessing
import warnings
warnings.filterwarnings("ignore")
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import sklearn as sk
import math
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
import os
from sklearn.preprocessing import scale


def make_subf(root_dir, sub_folder):
    plot_dir = os.path.join(root_dir, sub_folder)  # make a folder for the auto-report
    from errno import EEXIST
    from os import makedirs, path

    try:
        os.mkdir(plot_dir)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(plot_dir):
            pass
        else:
            raise

    return plot_dir

#get the baseline by ALS algorithm
def baseline_als(y,lam,p,niter=10):
    s  = len(y)
    # assemble difference matrix
    D0 = sparse.eye( s )
    d1 = [np.ones( s-1 ) * -2]
    D1 = sparse.diags( d1, [-1] )
    d2 = [ np.ones( s-2 ) * 1]
    D2 = sparse.diags( d2, [-2] )
    D  = D0 + D2 + D1
    w  = np.ones( s )
    for i in range( niter ):
        W = sparse.diags( [w], [0] )
        Z =  W + lam*D.dot( D.transpose() )
        z = spsolve( Z, w*y )
        w = p * (y > z) + (1-p) * (y < z)
    return z
def bl_als_nir(y,lam,p,niter=10):
    s  = len(y)
    # assemble difference matrix
    D0 = sparse.eye( s )
    d1 = [np.ones( s-1 ) * -2]
    D1 = sparse.diags( d1, [-1] )
    d2 = [ np.ones( s-2 ) * 1]
    D2 = sparse.diags( d2, [-2] )
    D  = D0 + D2 + D1
    w  = np.ones( s )
    for i in range( niter ):
        W = sparse.diags( [w], [0] )
        Z =  W + lam*D.dot( D.transpose() )
        z = spsolve( Z, w*y )
        w = p * (y < z) + (1-p) * (y > z)
    return z

def als(args):
    chunk=args[0]
    lam=args[1]
    p=args[2]
    chunk = [chunk[i, :] - baseline_als(chunk[i, :], lam, p) for i in range(chunk.shape[0])]
    return chunk
# remove baseline from spectra
def remove_baseline(fSample,lam,p):
    lam=10**lam;
    p=p/1000;
    return [fSample[i,:]-baseline_als(fSample[i,:], lam, p) for i in range(fSample.shape[0])]


def remove_bl_par(fSample, lam, p):
    lam=10**lam;
    p=p/1000;
    # Chunks for the mapping (only a few chunks):
    chunks = [(sub_arr, lam,p) for sub_arr in np.array_split(fSample, multiprocessing.cpu_count())]

    pool = multiprocessing.Pool()
    individual_results = pool.map(als, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)



def preprocess(fspec,bl,savgol,scaling):
    #fspec=savgol_filter(spectra,ch1[0],ch1[1],0)
    if bl['remove_bl']:
        fspec=remove_baseline(fspec,bl['Lambda'],bl['p'])
    fspec=savgol_filter(fspec,savgol['Window'],savgol['Order'],savgol['Deriv'])
    if scaling==0:
        return  StandardScaler().fit_transform(fspec.T).T   #preprocessing.scale(fspec) #column- wise snv :0 mean 1 std
        # fspec=RobustScaler(quantile_range=(25, 75)).fit_transform(fspec) #
    elif scaling==1:
        fspec=sk.preprocessing.normalize(fspec, norm='max')
    elif scaling==2:
        fspec=sk.preprocessing.normalize(fspec, norm='l1')
    return fspec



def msc(input_data, reference=None):
    ''' Perform Multiplicative scatter correction'''
  #input_data=pd.Series(input_data)
    # mean centre correction
    for i in range(input_data.shape[0]):
        input_data[i,:] -= input_data[i,:].mean()

    # Get the reference spectrum. If not given, estimate it from the mean
    if reference is None:
        # Calculate mean
        ref = np.mean(input_data, axis=0)
    else:
        ref = reference

    # Define a new array and populate it with the corrected data
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Run regression
        fit = np.polyfit(ref, input_data[i,:], 1, full=True)
        # Apply correction
        data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0]

    return [data_msc, ref]



def emsc_basic(spectra):
    emsc = importr('EMSC')
    return emsc.EMSC(spectra)[0]
def emsc_poly(spectra, degree, replicates,reference=0):
    emsc = importr('EMSC')
    return emsc.EMSC(spectra, degree=degree, replicates=replicates)[0]



def vip(x, model):

    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_

    m, p = x.shape
    h = t.shape[1]

    vips = np.zeros([p])

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    ss = np.sum(s)

    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p*(s.T @ weight)/ss)

    return vips


def out_T2_Q( X_test, Y, outl_dir, title):
    df=pd.concat([Y,X_test], axis=1).dropna()
    Y=df.iloc[:,0]
    X_test=df.iloc[:,1:]

    df = pd.concat([Y, X_test], axis=1).dropna()
    Y = df.iloc[:, 0]
    X_test = df.iloc[:, 1:]

    # score, ncomp = ml.pls(X_train, Y_train, metric='neg_mean_squared_error', cv=5)
    ncomp=5
    # Define PLS object
    pls = PLSRegression(n_components=ncomp)
    # Fit data
    X2 = scale(X_test)
    Y2 = scale(Y)
    pls.fit(X2, Y2)
    # Get X scores
    T = pls.x_scores_
    # Get X loadings
    P = pls.x_loadings_
    # Calculate error array
    Err = X2 - np.dot(T, P.T)
    # Calculate Q-residuals (sum over the rows of the error array)
    Q = np.sum(Err ** 2, axis=1)
    # Calculate Hotelling's T-squared (note that data are normalised by default)
    Tsq = np.sum((pls.x_scores_ / np.std(pls.x_scores_, axis=0)) ** 2, axis=1)

    conf = 0.95
    from scipy.stats import f
    # Calculate confidence level for T-squared from the ppf of the F distribution
    Tsq_conf = f.ppf(q=conf, dfn=ncomp, \
                     dfd=X2.shape[0]) * ncomp * (X2.shape[0] - 1) / (X2.shape[0] - ncomp)
    # Estimate the confidence level for the Q-residuals
    i = np.max(Q) + 1
    while 1 - np.sum(Q > i) / np.sum(Q > 0) > conf:
        i -= 1

    Q_conf = i
    rms_dist = np.flip(np.argsort(np.sqrt(Q ** 2 + Tsq ** 2)), axis=0)
    # Sort calibration spectra according to descending RMS distance
    Xc = X2[rms_dist, :]
    Yc = Y2[rms_dist]

    max_outliers = sum(Tsq > Tsq_conf) + sum(Q > Q_conf)

    # Define empty mse array
    mse = np.zeros(max_outliers)
    for j in range(max_outliers):
        pls = PLSRegression(n_components=ncomp)
        pls.fit(Xc[j:, :], Yc[j:])
        y_cv = cross_val_predict(pls, Xc[j:, :], Yc[j:], cv=5)
        mse[j] = mean_squared_error(Yc[j:], y_cv)
    # Find the position of the minimum in the mse (excluding the zeros)
    msemin = np.where(mse == np.min(mse[np.nonzero(mse)]))[0][0]
    inds = rms_dist[:msemin]
    plotting.hot_T_Q(Tsq, Tsq_conf, Q, Q_conf, outl_dir, title, inds, X_test.index)
    return X_test.index[inds]


def outliers_old(X_df):

    X_df = X_df.dropna(axis=0)
    x = X_df
    merging_col = x.index.name

    x = StandardScaler().fit_transform(x)

    ##Conduct PCA##
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)

    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

    ##Create Final data frame with principal components and merging column##
    principalDf = pd.concat([principalDf, X_df.reset_index().iloc[:, 0]], axis=1)

    # ##Calculate Centroid##
    centroid = np.array(
        [[principalDf["principal component 1"].mean(), principalDf["principal component 2"].mean(), "centroid"]])
    centroid = pd.DataFrame(centroid, columns=["principal component 1", "principal component 2", merging_col])

    # ##Add Centroid to finalDf##
    centroid_Df = pd.concat([principalDf, centroid], axis=0)
    centroid_Df.reset_index()

    #####Calculate Distances from Each Point to Centroid#####
    distances = []  ##Define empty list
    ##Iterate through finalDf and perform distance calculations##
    for i in range(0, len(principalDf)):
        d = math.sqrt((float(centroid_Df.iloc[i][0]) - float(centroid_Df.iloc[len(centroid_Df) - 1][0])) ** 2
                      + (float(centroid_Df.iloc[i][1]) - float(centroid_Df.iloc[len(centroid_Df) - 1][1])) ** 2)
        distances.append(d)

    ##Convert Distance List to Data Frame##
    distance_df = pd.DataFrame(distances)
    distance_df = distance_df.rename(columns={0: "Distance"})

    ##Reset indices##
    distance_df = distance_df.loc[~distance_df.index.duplicated(keep='first')]
    principalDf = principalDf.loc[~principalDf.index.duplicated(keep='first')]

    # ##Concatenate Final Df to
    principalDf = pd.concat([principalDf, distance_df], axis=1)

    # Find best fit distribution
    best_fit_name, best_fit_params, best_fit_int = plotting.best_fit_distribution(principalDf["Distance"], 200)

    group_list = []
    for distance in principalDf["Distance"]:
        if distance > best_fit_int[1]:
            group_list.append("possible outlier")

        else:
            group_list.append("standard")

    principalDf["group"] = group_list

    ##construct dataframe that is equivalent to principalDf, but sorted by Distance column to rank outliers
    sorted_principalDf = principalDf.sort_values(by="Distance", ascending=False)
    sorted_principalDf = sorted_principalDf[sorted_principalDf['group'] == "possible outlier"]

    ##Gets list of top 10 outliers from sorted_principalDf
    if len(sorted_principalDf) > 10:
        ranked_outliers = list(sorted_principalDf[X_df.index.name].iloc[0:10, ])

    else:
        ranked_outliers = list(sorted_principalDf[X_df.index.name])

    ##Compile list of outliers basd on p_values in principalDf
    i = 0
    a = 0.05
    outliers = []
    for p in principalDf["group"]:
        if p == "possible outlier":
            i_outlier = X_df.index[i]
            # print(i_outlier)
            outliers.append(i_outlier)
        i = i + 1
    # print("Possiblie otutliers identified are: ", outliers)


    # hist_df = hist_df.drop(["p_val"],axis = 1)

    return outliers, ranked_outliers, principalDf, best_fit_int


def save_outlier_plots_old(X_df,plot_dir,title=''):
    all_outliers, ranked_outliers,  principalDf, best_fit_int = outliers_old(X_df)
    ##### Outlier Biplot ###############
    ###### Distance to Centroid #############
    hist_dir="{}/{}".format(plot_dir, title.replace(' ','_')+'_outliers_hist.jpg')
    plotting.plot_hist(principalDf, title, plot_dir=hist_dir, best_fit_int= best_fit_int)

    biplot_dir="{}/{}".format(plot_dir, title.replace(' ','_')+'_outliers_biplot.jpg')
    pc_spec_dir1="{}/{}".format(plot_dir, title.replace(' ','_')+'_outliers_pcN1.jpg')
    pc_spec_dir2="{}/{}".format(plot_dir, title.replace(' ','_')+'_outliers_pcN2.jpg')


    plotting.plot_biplot(principalDf, title, plot_dir=biplot_dir, best_fit_int=best_fit_int, pc_spec_dir1=pc_spec_dir1, pc_spec_dir2=pc_spec_dir2)

    return all_outliers, ranked_outliers


def corr_rep(X_df,Y_df,raw_spec_dir,outl_dir, title, remove_out=False):

    wl = X_df.columns[1:]

    all_outliers, ranked_outliers=[],[]
    title1=title+' Before EMSC'
    all_outliers, ranked_outliers = save_outlier_plots_new(X_df.iloc[:, 1:], outl_dir, title=title1 )

    plotting.plot_spectra(wl, X_df.iloc[:, 1:].to_numpy(), 'Raw ' + title + ' Spectra', colors="spectral",
                          alloutliers=X_df.loc[all_outliers, :].iloc[:, 1:].to_numpy(),
                          rankedoutliers=X_df.loc[ranked_outliers, :].iloc[:, 1:].to_numpy(),
                          save_fig=True, file_dir=raw_spec_dir)

    spec_dir = "{}/{}".format(outl_dir, title1.replace(' ', '_') + '_outliers_spec.jpg')
    plotting.plot_spectra(wl, X_df.loc[all_outliers, :].iloc[:, 1:].to_numpy(), title + ' Outliers Spectra Before EMSC',
                          colors="spectral", alloutliers=X_df.loc[all_outliers, :].iloc[:, 1:].to_numpy(),
                          rankedoutliers=X_df.loc[ranked_outliers, :].iloc[:, 1:].to_numpy(),
                          save_fig=True, file_dir=spec_dir)

    if remove_out:

        X_df = X_df.drop(all_outliers)
        Y_df = Y_df.drop(all_outliers)
    # do replicate correction
    samples = X_df.loc[:, 'Replicate Group ID']
    for sample in samples.unique():
        indx = samples[samples == sample].index
        scans=X_df.loc[indx, :].iloc[:, 1:]
        X_df.loc[indx, :].iloc[:, 1:] = emsc_poly(scans.to_numpy(), degree=6,replicates=X_df.loc[indx, 'Replicate Group ID'],reference=scans.mean())

    title2 = title + ' After EMSC'
    all_outliers_after, ranked_outliers_after = save_outlier_plots_new(X_df.iloc[:, 1:], plot_dir=outl_dir, title=title2)

    spec_dir="{}/{}".format(outl_dir, title2.replace(' ','_')+'_outliers_spec.jpg')
    plotting.plot_spectra(wl, X_df.loc[all_outliers_after, :].iloc[:, 1:].to_numpy(), title + ' Outliers Spectra After EMSC',
                          colors="spectral",alloutliers=X_df.loc[all_outliers_after, :].iloc[:, 1:].to_numpy(),
                          rankedoutliers=X_df.loc[ranked_outliers_after, :].iloc[:, 1:].to_numpy(),
                          save_fig=True, file_dir=spec_dir)

    rep_ids = X_df['Replicate Group ID']
    X_df.drop('Replicate Group ID',axis=1, inplace=True)
    Y_df.drop('Replicate Group ID', axis=1,inplace=True)


    return X_df, Y_df, rep_ids, all_outliers,ranked_outliers


def simca_prep(spectra, window, order, deriv):
    dspectra = preprocess(spectra, {'remove_bl': 0},
                               {'Window': window, 'Order': order, 'Deriv': deriv}, 5)
    N=dspectra.shape[1]

    return  StandardScaler().fit_transform(dspectra.T).T * np.sqrt((N - 1) / N) #simca SNV ----along the wavenumbers - raw-wise


def outlier_to_string(outlier_list):
    # returns comma separated string of the outlier file names
    string = outlier_list[0]
    newline = 0
    for file in outlier_list[1:]:
        string = string + ", "
        if newline % 3 == 0:
            string += "\n"
        string += file
        newline += 1

    return string


def outliers_new(X_df):
    """Returns dataframe of principal components, lists of outliers, p-values.

    arguments - 1 item
    X_df: spectra data as an excel file
    returned by function - 4 items

    1. list_of_list_outliers: each item is a list of the index positions of outliers for a pair of PCs
    2. list_of_list_ranked_outliers: each item is a list of the index positions of outliers for a pair of PCs, but sorted according to the calculated distance.
    3. principalDf: dataframe of all principal components calculated form X_df, including distances and outlier labels
    4. list_best_fit_int: a list of the p-value for the right side of the 95% confidence interval

    Notes:
    The function select_df is used for pc selection; it currently requires declaration of global variables
    """
    X_df = X_df.dropna(axis=0)

    list_best_fit_int = []
    list_of_list_ranked_outliers = []
    list_of_list_outliers = []
    x_df_index = X_df.index.values.tolist()

    x = X_df.copy()
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=6)
    principalComponents = pca.fit_transform(x)

    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2',
                                                                  'principal component 3', 'principal component 4',
                                                                  'principal component 5', 'principal component 6'])
    n_PC_components = principalComponents.shape[1]
    var_explained = pca.explained_variance_ratio_

    # run calculate_distances() on the main principal df for the PC1 vs PC2 combination
    principalDf = calculate_distances(principalDf,
                                      "Distance: PC1&PC2", "principal component 1", "principal component 2")

    # use the best_fit_distribution method to calculate the p-value on the distances between PC1 and PC2
    best_fit_name_1_2, best_fit_params_1_2, best_fit_int_1_2 = plotting.best_fit_distribution(
        principalDf["Distance: PC1&PC2"], 200)

    # run create_outliers() on the prindcipalDf and based on the distances between PC1 and PC2
    principalDf = create_outlier_group(principalDf, best_fit_int_1_2[1], "possible outlier",
                                       "standard", "Distance: PC1&PC2", "group: PC1&PC2")
    # append to output lists; only want first element of best_fit_int (right side CI)
    list_best_fit_int.append(best_fit_int_1_2[1])

    list_of_list_ranked_outliers.append(rank_outliers(principalDf, "Distance: PC1&PC2",
                                                      "group: PC1&PC2", "possible outlier", x_df_index))

    list_of_list_outliers.append(plain_outliers(principalDf, "group: PC1&PC2",
                                                "possible outlier", x_df_index))

    # run calculate_distances() on the main principal df for the PC3 vs PC4 combination
    principalDf = calculate_distances(principalDf, "Distance: PC3&PC4", "principal component 3",
                                      "principal component 4")

    # use the best_fit_distribution method to calculate the p-value on the distances between PC3 and PC4
    best_fit_name_3_4, best_fit_params_3_4, best_fit_int_3_4 = plotting.best_fit_distribution(
        principalDf["Distance: PC3&PC4"], 200)

    # run create_outliers() on the prindcipalDf and based on the distances between PC3 and PC4
    principalDf = create_outlier_group(principalDf, best_fit_int_3_4[1], "possible outlier",
                                       "standard", "Distance: PC3&PC4", "group: PC3&PC4")

    # append to output lists; want only second element of best_fit_int (right side CI)
    list_best_fit_int.append(best_fit_int_3_4[1])
    list_of_list_ranked_outliers.append(rank_outliers(principalDf, "Distance: PC3&PC4",
                                                      "group: PC3&PC4", "possible outlier", x_df_index))
    list_of_list_outliers.append(plain_outliers(principalDf, "group: PC3&PC4",
                                                "possible outlier", x_df_index))

    # run calculate_distances() on the main principal df for the PC5 vs PC6 combination
    principalDf = calculate_distances(principalDf,
                                      "Distance: PC5&PC6", "principal component 5", "principal component 6")

    # use the best_fit_distribution method to calculate the p-value on the distances between PC5 and PC6
    best_fit_name_5_6, best_fit_params_5_6, best_fit_int_5_6 = plotting.best_fit_distribution(
        principalDf["Distance: PC5&PC6"], 200)

    # run create_outliers() on the prindcipalDf and based on the distances between PC5 and PC6
    principalDf = create_outlier_group(principalDf, best_fit_int_5_6[1], "possible outlier",
                                       "standard", "Distance: PC5&PC6", "group: PC5&PC6")
    # append to output lists; only want first element of best_fit_int (right side CI)
    list_best_fit_int.append(best_fit_int_5_6[1])
    list_of_list_ranked_outliers.append(rank_outliers(principalDf, "Distance: PC5&PC6",
                                                      "group: PC5&PC6", "possible outlier", x_df_index))
    list_of_list_outliers.append(plain_outliers(principalDf, "group: PC5&PC6",
                                                "possible outlier", x_df_index))

    return list_of_list_outliers, list_of_list_ranked_outliers, principalDf, list_best_fit_int, var_explained, n_PC_components

def calculate_distances(df, new_col, col_1, col_2):
    """calculate_distances runs a lambda on each row and appends to the dataframe"""
    df[new_col] = df.apply(lambda x: math.sqrt((x[col_1] - df[col_1].mean()) ** 2 + (x[col_2] - df[col_2].mean()) ** 2),
                           axis=1)
    return df

def create_outlier_group(df, best_fit_int, outlr_val, stndrd_val, column_criteria, new_column):
    """create_outlier_group() creates a list of strings based on the outlier criteria 'best_fit_int'"""
    group_list = []
    [group_list.append(outlr_val) if x > best_fit_int else group_list.append(stndrd_val) for x in df[column_criteria]]
    df[new_column] = group_list
    return df

def rank_outliers(df, sort_key, group_col, criteria, x_df_index):
    """ranked_outliers(): sorts df by distance and finds index values for the outliers to find the corresponding row labels in X_df"""
    sorted_df = df.sort_values(by=sort_key, ascending=False)
    sorted_df = sorted_df[sorted_df[group_col] == criteria]
    list_ranked = list(sorted_df.index[0:10]) if len(sorted_df) > 10 else list(sorted_df.index)
    list_indices = []
    [list_indices.append(x_df_index[item]) for item in list_ranked]
    return list_indices

def plain_outliers(df, group_col, criteria, x_df_index):
    """finds index values for the outliers to find the corresponding row labels in X_df"""
    list_outliers = df[df[group_col] == criteria].index.tolist()
    list_indices = []
    [list_indices.append(x_df_index[item]) for item in list_outliers]
    return list_indices

def save_outlier_plots_new(X_df, plot_dir, title=''):
    """runs biplot and histogram

    arguments - 3 items

    1. X_df: spectra data as an excel file
    2. plot_dir: directory where plots are be saved; agora attribute
    3. title: title for plots; agora attribute

    returns - 2 items

    1. all_outliers: first item of all_outliers list reuturned by outliers function
    2. ranked_outliers: first item of ranked_outliers list reuturned by outliers function

    Note:
    Hard-coding for items selected from outliers lists and distance column selection for histogram
    """
    list_of_list_all_outliers, list_of_list_ranked_outliers, principalDf, list_best_fit_int, \
    var_explained, n_PC_components = outliers_new(X_df)

    ##### Outlier Biplot ###############
    ###### Distance to Centroid #############
    list_distance_col = []
    #Hard coding for distance column selection for histogram function
    list_distance_col.append(principalDf.columns[6])
    list_distance_col.append(principalDf.columns[8])
    list_distance_col.append(principalDf.columns[10])
    hist_dir = "{}/{}".format(plot_dir, title.replace(' ', '_') + '_outliers_hist.jpg')
    for item in np.arange(0, 3):
        plotting.plot_hist_new(principalDf, title, hist_dir, list_best_fit_int[item], list_distance_col[item])
    plotting.biplot_new(principalDf, X_df, 1, 2, title, plot_dir)
    plotting.plot_PCscans(principalDf, X_df, 1, 2, title, plot_dir, plot_dir)
    exp_var_dir = "{}/{}".format(plot_dir, title.replace(' ', '_') + '_PCcomp_vs_var_exp.jpg')
    plotting.plot_ncomps_vs_exp_var(n_PC_components, var_explained, exp_var_dir)

    # set return values to first elements of output lists; hard coding for selection of first item
    all_outliers = list_of_list_all_outliers[0]
    ranked_outliers = list_of_list_ranked_outliers[0]

    return all_outliers, ranked_outliers