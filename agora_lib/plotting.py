import datetime
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from matplotlib import rc
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from scipy.stats import sem, t
from sklearn.metrics import mean_squared_error, r2_score

from agora_lib import prep

matplotlib_axes_logger.setLevel('ERROR')
import numpy as np

font = 18
font_model_sec = 22
font_desc_stat_sec = 22
fig_size = (9, 5)


def conf_interval(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)

    start = m - h
    end = m + h
    print(start)

    return [np.round(start, 3), np.round(end, 3)]


def reverse_colourmap(cmap, name='my_cmap_r'):
    reverse = []
    k = []

    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1 - t[0], t[2], t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k, reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r


def pred_err_plot(y_valid, y_pred, file_dir, title, attr_name, accepted_err=0.1):
    diff = y_pred - y_valid

    n = y_valid.shape[0]
    rel_error = abs(np.round(diff / y_valid, 6))
    fiq, ax = plt.subplots(figsize=fig_size)
    cmap = plt.get_cmap('seismic', n)
    df = pd.DataFrame(data=rel_error, index=np.arange(n), columns=['RelErr']).sort_values(by='RelErr',
                                                                                          ascending=False).reset_index()
    df.rename(columns={'index': 'meas_arg'}, inplace=True)
    for i in np.argsort(y_valid):
        color_idx = df[df.meas_arg == i].index  # get a color index based on rel error rank
        ax.scatter(y_valid[i], diff[i], s=50, c=cmap(color_idx), edgecolors='black')
    x0 = 0
    y0 = 0
    y_max = diff.max() * 1.05
    x_max0 = y_valid.max()
    diff_max0 = np.sign(y_max) * x_max0 * accepted_err / (accepted_err * np.sign(y_max) + 1)

    max_x0x1 = [x0, x_max0]
    max_y0y1 = [y0, diff_max0]
    plt.plot(max_x0x1, max_y0y1, '--', c='r')

    y_min = diff.min() * 1.05
    x_max = y_valid.max()
    diff_max = np.sign(y_min) * x_max * accepted_err / (accepted_err * np.sign(y_min) + 1)

    min_x0x1 = [x0, x_max]
    min_y01y1 = [y0, diff_max]
    plt.plot(min_x0x1, min_y01y1, '--', c='r')

    ax.fill_between(max_x0x1, max_y0y1, y_max, facecolor="orange",  # The fill color
                    color='red',  # The outline color
                    alpha=0.1, label="Error $>$ %d%%" % (accepted_err * 100))
    ax.fill_between(min_x0x1, min_y01y1, y_min, facecolor="orange",  # The fill color
                    color='red',  # The outline color
                    alpha=0.1)
    #plt.title(title, fontsize=font)
    plt.ylim(y_min, y_max)
    plt.xlim(x0, y_valid.max())
    plt.xlabel('Measurement', fontsize=font_model_sec)
    plt.ylabel('Absolute Error', fontsize=font_model_sec)

    vmin = rel_error.min() * 100

    vmax = rel_error.max() * 100

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #plt.grid('on', alpha=0.25)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.set_title("Error %", fontsize=font_model_sec, position=(1, -0.1))
    ax.legend(loc='best', fancybox=True, shadow=True, ncol=1, fontsize='medium')
    rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': font_model_sec, 'weight': 'light'})

    plt.savefig(file_dir, bbox_inches="tight")
    plt.close()


def corr_plot(y_pred, y_test, attr, file_dir):
    eps = np.finfo(np.float32).eps
    r2 = r2_score(y_test, y_pred)
    z = np.polyfit(np.float32(y_test), y_pred, 1, rcond=len(y_test) * eps)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.scatter(y_pred, y_test, c='red', edgecolors='k')
    if z[1] > 0:
        label = "%5.2f $x$+ %5.2f"
    else:
        label = "%5.2f $x$ %5.2f"

    ax.plot(z[1] + z[0] * y_test, y_test, c='blue', linewidth=1, label=label % (z[0], z[1]))
    ax.plot(y_test, y_test, color='green', linewidth=1)
    plt.xlabel('Predicted', fontsize=font_model_sec)
    plt.ylabel('Measured', fontsize=font_model_sec)
    title = 'Correlation for ' + attr;
    plt.title(title)
    ax.legend(fontsize=font_model_sec, loc="lower right")
    xx = 0.18
    plt.figtext(xx, 0.75, 'R$^{2}$ : %5.3f' % r2, fontsize=font_model_sec)

    rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': font_model_sec, 'weight': 'light'})

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
    ):  # ax.get_xticklabels() + ax.get_yticklabels()
        item.set_fontsize(font_model_sec)

    for item in ax.get_xticklabels():
        item.set_fontsize(font_model_sec)
    for item in ax.get_yticklabels():
        item.set_fontsize(font_model_sec)

    plt.tight_layout()

    plt.rcParams['font.size'] = font_model_sec

    fig.savefig(file_dir, bbox_inches="tight");
    plt.close()

    return z


def corr_plot_univ(y_pred, y_test, fitness_cv, fitness_test, title, file_dir, val_set_given, fitness_valid=0):
    eps = np.finfo(np.float32).eps
    r2 = r2_score(y_test, y_pred)
    z = np.polyfit(np.float32(y_test.flatten()), y_pred.flatten(), 1, rcond=len(y_test) * eps)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.scatter(y_pred, y_test, c='red', edgecolors='k')
    if z[1] > 0:
        label = "%5.2f $x$+ %5.2f"
    else:
        label = "%5.2f $x$ %5.2f"

    ax.plot(z[1] + z[0] * y_test, y_test, c='blue', linewidth=1, label=label % (z[0], z[1]))
    ax.plot(y_test, y_test, color='green', linewidth=1)
    plt.xlabel('Predicted', fontsize=font_model_sec)
    plt.ylabel('Measured', fontsize=font_model_sec)
    title = title;
    #plt.title(title)
    ax.legend(fontsize=font_model_sec, loc="lower right")
    xx = 0.18
    plt.figtext(xx, 0.75, 'R$^{2}$ : %5.3f' % r2, fontsize=font_model_sec)
    plt.figtext(xx, 0.7, 'RMSE CV: %5.3f' % fitness_cv, fontsize=font_model_sec)

    # if 'SIMCA' in title:
    #     y = 0.65
    #     if self.val_set_given:
    #         plt.figtext(xx, y, 'RMSEP : %5.3f' % .rmsep, fontsize=font)
    #     else:
    #         plt.figtext(xx, y, 'RMSEE : %5.3f' % self.rmsee, fontsize=font)

    plt.figtext(xx, 0.65, 'RMSE Test: %5.3f' % fitness_test, fontsize=font_model_sec)

    y = 0.6
    if val_set_given:
        plt.figtext(xx, y, 'RMSE Valid: %5.3f' % fitness_valid, fontsize=font_model_sec)
        xx -= 0.05
    #  plt.figtext(xx,y, '95% CI :' + str(conf_interval(abs(y_pred.flatten()-y_test.flatten()))), fontsize=font)

    rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': font_model_sec, 'weight': 'light'})

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
    ):  # ax.get_xticklabels() + ax.get_yticklabels()
        item.set_fontsize(font_model_sec)

    for item in ax.get_xticklabels():
        item.set_fontsize(font_model_sec)
    for item in ax.get_yticklabels():
        item.set_fontsize(font_model_sec)

    plt.tight_layout()

    plt.rcParams['font.size'] = font_model_sec

    fig.savefig(file_dir, bbox_inches="tight");
    plt.close()

    return z


def corr_plot_patent(y_pred, y_test, attr_name, title, file_dir):
    eps = np.finfo(np.float32).eps
    y_test = y_test / max(y_test)
    y_pred = y_pred / max(y_pred)
    r2 = r2_score(y_test, y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    z = np.polyfit(np.float32(y_test.flatten()), y_pred.flatten(), 1, rcond=len(y_test) * eps)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.scatter(y_pred, y_test, c='black', edgecolors='k')

    ax.plot(z[1] + z[0] * y_test, y_test, c='black', linestyle='-.', linewidth=1)  # , label=label % (z[0],z[1]))
    ax.grid(False)

    plt.xlabel('Predicted')
    plt.ylabel('Measured')
    plt.title(title)
    #  ax.legend(fontsize=font,loc="lower right")
    xx = 0.18
    plt.figtext(xx, 0.75, 'R$^{2}$ : %5.3f' % r2, fontsize=font_model_sec)
    plt.figtext(xx, 0.7, 'RMSE : %5.3f' % rmse, fontsize=font_model_sec)

    rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': font_model_sec, 'weight': 'light'})

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
    ):  # ax.get_xticklabels() + ax.get_yticklabels()
        item.set_fontsize(font_model_sec)

    for item in ax.get_xticklabels():
        item.set_fontsize(font_model_sec)
    for item in ax.get_yticklabels():
        item.set_fontsize(font_model_sec)

    plt.tight_layout()
    attr_name = attr_name.replace('/)', '')
    attr_name = attr_name.replace('/', '')

    plot_dir = '{}/' + attr_name + "_patent_" + str(datetime.datetime.now()) + '.png'

    plt.rcParams['font.size'] = font_model_sec

    fig.savefig(plot_dir.format(file_dir), bbox_inches="tight");
    plt.close()


def plot_spectra_patent(wl, spectra, title, name, save_fig=False, file_dir=os.getcwd()):
    n = spectra.shape[0]

    colors = plt.cm.seismic(np.linspace(0, 1, n))

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    for i in range(n):
        ax.plot(wl, spectra[i, :].T, color=colors[i])

    plt.xlabel('Wavenumber')
    plt.ylabel('Intensity')

    plt.xticks([], [])
    plt.yticks([], [])
    plt.title(title)

    rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': font, 'weight': 'light'})

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
    ):  # ax.get_xticklabels() + ax.get_yticklabels()
        item.set_fontsize(font)

    for item in ax.get_xticklabels():
        item.set_fontsize(font)
    for item in ax.get_yticklabels():
        item.set_fontsize(font)

    # ax.ticklabel_format(axis='y', style='sci', scilimits=(-6, 6), useMathText=True)
    plt.tight_layout()
    if save_fig:
        plt.savefig(file_dir + "/" + "aspectra_" + name + ".jpg", bbox_inches="tight")
    plt.close()


def plot_spectra(wl, spectra, title, font_size=font, colors="binary", alloutliers=None, rankedoutliers=None, save_fig=False, file_dir=os.getcwd()):

    n = spectra.shape[0]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    if colors=="binary" or rankedoutliers is None:
        if n > 1:
            colors = plt.cm.binary(np.linspace(0, 1, n + 1))
            for i in range(n):
                plt.plot(wl, spectra[i, :].T, color=colors[i])
        elif n == 1:
            plt.plot(wl, spectra.T, color='black')
        else:
            title = 'No Outliers Detected'
    else:
        if rankedoutliers.shape[0] < 11:
            top_ten_outl = rankedoutliers
        else:
            top_ten_outl = rankedoutliers[0:10,:]
        if n > 1:
            for i in range(n):
                if np.isin(spectra[i, :], top_ten_outl).all():
                    colors = plt.cm.nipy_spectral(np.linspace(0, 1, n + 1))
                    plt.plot(wl, spectra[i, :].T, color=colors[i], label=i)
                elif np.isin(spectra[i, :], alloutliers).all():
                    colors = plt.cm.nipy_spectral(np.linspace(0, 1, n + 1))
                    plt.plot(wl, spectra[i, :].T, color=colors[i])
                else:
                    colors = plt.cm.binary(np.linspace(0, 1, n + 1))
                    plt.plot(wl, spectra[i, :].T, color=colors[i])
        elif n == 1:
            colors = plt.cm.nipy_spectral(np.linspace(0, 1, n + 1))
            plt.plot(wl, spectra.T, color=colors[0])
        else:
            title = 'No Outliers Detected'
    plt.xlabel('Wavenumber')
    plt.ylabel('Intensity')
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) < 11:
        elem_handles = len(handles)
    else:
        elem_handles = 10
    ax.legend(handles[0:elem_handles], np.arange(0, elem_handles), loc='upper center', bbox_to_anchor=(0.5, -0.2),
               title='Top 10 Outliers', fancybox=True, shadow=True, ncol=5, fontsize=10, title_fontsize=12)

    plt.title(title)

    rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': font_size, 'weight': 'light'})

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):  # ax.get_xticklabels() + ax.get_yticklabels()
        item.set_fontsize(font)

    for item in ax.get_xticklabels():
        item.set_fontsize(font_size)
    for item in ax.get_yticklabels():
        item.set_fontsize(font_size)

    ax.ticklabel_format(axis='y', style='sci', scilimits=(-6, 6), useMathText=True)
    plt.tight_layout()
    if save_fig:
        fig.savefig(file_dir, bbox_inches="tight");
        plt.close()


def infl_plot(x1, y1, attr_name, plot_dir):
    lm = sm.OLS(y1, sm.add_constant(x1)).fit()
    lm.predict()
    print("The rsquared values is " + str(lm.rsquared))
    fig, ax = plt.subplots(figsize=(9, 5))
    fig = sm.graphics.influence_plot(lm, alpha=0.05, ax=ax, criterion="cooks")
    #title = 'Influence Plot: ' + attr_name
   # plt.title(title, fontsize=font)
    plt.figtext(0.18, 0.75, 'R$^{2}$ (marked samples dropped): %5.3f' % lm.rsquared, fontsize=font)

    rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': font, 'weight': 'light'})

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
    ):  # ax.get_xticklabels() + ax.get_yticklabels()
        item.set_fontsize(font)

    for item in ax.get_xticklabels():
        item.set_fontsize(font)
    for item in ax.get_yticklabels():
        item.set_fontsize(font)

    plt.savefig(plot_dir, bbox_inches="tight")  #
    plt.close()


def hot_T_Q(Tsq, Tsq_conf, Q, Q_conf, plot_dir, title, inds, labels):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    t_ind = np.where(Tsq > Tsq_conf)[0]
    q_ind = np.where(Q > Q_conf)[0]

    colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(t_ind) + len(q_ind) + 1))
    t_ind = np.where(Tsq > Tsq_conf)[0]
    q_ind = np.where(Q > Q_conf)[0]
    k = 1
    if len(inds) > 30:
        inds = inds[:30]

    for i in t_ind:
        if i in inds:
            label = str(labels[i]) + ' - remove'

            plt.plot(Tsq[i], Q[i], 'o', color=colors[k], label=label)
        else:
            # label=str(labels[i])
            plt.plot(Tsq[i], Q[i], 'o', color=colors[k])  # , label=label)

        k += 1
    plt.plot(Tsq[Tsq < Tsq_conf], Q[Tsq < Tsq_conf], 'o', color=colors[0])
    for j in q_ind:
        if j in inds:
            label = str(labels[j]) + ' - remove'

            plt.plot(Tsq[j], Q[j], 'o', color=colors[k], label=label)
        else:
            # label=str(labels[j])
            plt.plot(Tsq[j], Q[j], 'o', color=colors[k])  # , label=label)

        k += 1
    plt.legend(bbox_to_anchor=(1.01, 1))

    plt.plot([Tsq_conf, Tsq_conf], [plt.axis()[2], plt.axis()[3]], '--')
    plt.plot([plt.axis()[0], plt.axis()[1]], [Q_conf, Q_conf], '--')
    plt.xlabel("Hotelling's T-squared", fontsize=18)
    plt.ylabel('Q residuals', fontsize=18)
    plt.title('Outliers detected. ' + title, fontsize=18)

    rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': font, 'weight': 'light'})

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
    ):  # ax.get_xticklabels() + ax.get_yticklabels()
        item.set_fontsize(font)

    for item in ax.get_xticklabels():
        item.set_fontsize(font)
    for item in ax.get_yticklabels():
        item.set_fontsize(font)

    plt.savefig(plot_dir, bbox_inches="tight")
    plt.close()


# plt.legend(n_users.index,bbox_to_anchor=(2, 1.05),fontsize=20)
def plot_biplot(principalDf, title, plot_dir, best_fit_int, pc_spec_dir1, pc_spec_dir2):
    ##Output biplot of Principal Components##
    x_pca = np.array(principalDf["principal component 1"])
    y_pca = np.array(principalDf["principal component 2"])

    radius = best_fit_int[1]
    circle = plt.Circle((0, 0), radius, color='gray', fill=False)

    fig = plt.figure(figsize=fig_size, dpi=300)
    ax = fig.add_subplot(111)

    # plot points inside distribution's width
    outl = principalDf[principalDf["group"] == "possible outlier"].iloc[:, 2]
    outx = x_pca[principalDf["group"] == "possible outlier"]
    outy = y_pca[principalDf["group"] == "possible outlier"]

    colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(outx) + 1))

    ax.scatter(x_pca[principalDf["group"] == "standard"], y_pca[principalDf["group"] == "standard"], marker="s",
               color=colors[0])
    # color="#2e91be")
    # plot points outside distribution's width

    if len(outl) > 30:
        for i in range(len(outl)):
            #     ax.scatter(outx[i], outy[i], marker="d", color=colors[i+1], label=outl.iloc[i]) #"#d46f9f"
            # for i in np.arange(20, len(outl),1):
            ax.scatter(outx[i], outy[i], marker="d", color=colors[i + 1])  # "#d46f9f"
    else:
        for i in range(len(outl)):
            ax.scatter(outx[i], outy[i], marker="d", color=colors[i + 1], label=outl.iloc[i])  # "#d46f9f"
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.125),
                  fancybox=True, shadow=True, ncol=1)
    ax.add_artist(circle)

    plt.title(title, fontsize=font_desc_stat_sec)
    plt.xlabel('PC1', fontsize=font_desc_stat_sec)
    plt.ylabel('PC2', fontsize=font_desc_stat_sec)

    rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': font_desc_stat_sec, 'weight': 'light'})

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
    ):  # ax.get_xticklabels() + ax.get_yticklabels()
        item.set_fontsize(font_desc_stat_sec)

    for item in ax.get_xticklabels():
        item.set_fontsize(font_desc_stat_sec)
    for item in ax.get_yticklabels():
        item.set_fontsize(font_desc_stat_sec)

    fig.savefig(plot_dir, bbox_inches="tight");
    plt.close()


# def residuals_plot():
#     a = sm_fr.standard_resid[abs(sm_fr.standard_resid) > 2].index
#
#     f, axs = plt.subplots(8, 1, figsize=(6, 16), dpi=200)
#     cs = plt.cm.nipy_spectral(np.linspace(0, 1, len(a)))
#     i = 0
#     for col in sm_fr.columns:
#
#         axs[i].scatter(range(64), sm_fr[col])
#         k = 0
#         for ind in a:
#             axs[i].scatter(ind, sm_fr[col].loc[ind], color=cs[k], label=ind)
#             k += 1
#         axs[i].set_title(col)
#         axs[i].legend(loc='best')
#         i += 1
#     f.tight_layout()
#     plt.savefig('res.png', bbox_inches="tight")
#     plt.show()


def plot_hist(histogram_dataframe, h_title, plot_dir, best_fit_int):
    fig, ax = plt.subplots(figsize=fig_size)

    rwidth = 1.0
    edgecolor = 'k'

    n, bins, patches = ax.hist(histogram_dataframe.Distance, bins='auto', edgecolor=edgecolor, rwidth=rwidth)

    for i in range(len(patches)):
        if bins[i] < best_fit_int[1]:
            patches[i].set_facecolor('b')
        else:
            patches[i].set_facecolor('r')

    plt.xlabel('Euclidean Distance from Centroid', fontsize=font_desc_stat_sec)
    plt.ylabel('Total Count', fontsize=font_desc_stat_sec)
    plt.title('Distribution of Distances from Centroid: ' + h_title, fontsize=font_desc_stat_sec)

    # The import was required here; red_patch and blue_patch are 'Proxy artists' for the legend
    import matplotlib.patches as patches
    red_patch = patches.Patch(color='red', label='Possible outliers')
    blue_patch = patches.Patch(color='blue', label='Standard')
    ax.legend(handles=[red_patch, blue_patch])
    ax.grid(axis='y', alpha=0.25)

    rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': font_desc_stat_sec, 'weight': 'light'})

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
    ):
        item.set_fontsize(font_desc_stat_sec)

    for item in ax.get_xticklabels():
        item.set_fontsize(font_desc_stat_sec)
    for item in ax.get_yticklabels():
        item.set_fontsize(font_desc_stat_sec)

    plt.savefig(plot_dir, bbox_inches="tight")
    plt.close()


# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [
        st.norm, st.poisson
    ]
    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

                    ## Calculate 95% Confidence Interval for each distribution
                    if best_distribution.name == "norm":
                        conf_int = best_distribution.interval(0.95, loc=data.mean(), scale=data.std())

                    elif best_distribution.name == "poisson":
                        conf_int = best_distribution.interval(0.95, loc=data.mean(), scale=0)

        except Exception:
            pass

    return (best_distribution.name, best_params, conf_int)


def vip_bar(wl, coeffs, attr_name, file_dir):
    y = np.linspace(round(coeffs.min(), 1), round(coeffs.max(), 1), 10)
    y = np.round(y, 2)

    idx = np.argsort(coeffs)[::-1]
    above_1 = np.where(coeffs[idx] > 1)[0]

    fig, ax = plt.subplots(figsize=(30, 10))
    x = np.arange(wl.shape[0])
    plt.bar(x, coeffs[idx])
    plt.plot([plt.axis()[0], plt.axis()[1]], [1, 1], '--', c='r')
    if len(above_1)>0:
        plt.plot([x[above_1[-1]], x[above_1[-1]]], [plt.axis()[2], plt.axis()[3]], '--', c='r')
    plt.xticks(x[::20], wl[idx].astype(int)[::20], rotation=90, fontsize=font)
    plt.yticks(y, y, fontsize=font)
    #plt.title('VIP Plot: ' + attr_name)
    plt.ylim(coeffs.min(), coeffs.max())


    rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': font, 'weight': 'light'})

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
    ):  # ax.get_xticklabels() + ax.get_yticklabels()
        item.set_fontsize(font)

    for item in ax.get_xticklabels():
        item.set_fontsize(font)
    for item in ax.get_yticklabels():
        item.set_fontsize(font)
    plt.savefig(file_dir, bbox_inches="tight")
    plt.close()


def vip_plot2(wl, x_df, coeffs, attr_name, file_dir):
    n = x_df.shape[1]
    x = wl
    cmap = plt.get_cmap('binary', n)

    fig, ax = plt.subplots(figsize=fig_size)
    # ax1 = fig.add_axes([0.10, 0.10, 0.70, 0.85])

    for i in range(n):
        ax.scatter(x[i], x_df[1, i], c=cmap(np.argsort(coeffs)[i]))

    vmin = coeffs.min()
    vmax = coeffs.max()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax)
    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Intensity')
    #ax.set_title("VIP: " + attr_name)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
    ):
        item.set_fontsize(font)
    for item in ax.get_xticklabels():
        item.set_fontsize(font)
    for item in ax.get_yticklabels():
        item.set_fontsize(font)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-6, 6), useMathText=True)
    plt.tight_layout()
    plt.savefig(file_dir, bbox_inches="tight");
    plt.close()


def vip_plot(wl, x_df, coeffs, attr_name, file_dir):
    fig, ax = plt.subplots(figsize=fig_size, dpi=300)
    # create color settings (red, green, blue, alpha) for each data point, using t as transparent alpha value
    dotcolors = [(0.2, 0.4, 0.6, (a / max(coeffs)) ** 2) for a in coeffs]  ##Scaled so opacity range is between 0 and 1
    for scan in x_df:
        plt.scatter(np.array(wl), scan.T, c=dotcolors, s=0.5)
    plt.xlabel('Wavenumber')
    plt.ylabel('Intensity')
    plt.title(attr_name + " Variable Importance")
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
    ):
        item.set_fontsize(15)
    for item in ax.get_xticklabels():
        item.set_fontsize(25)
    for item in ax.get_yticklabels():
        item.set_fontsize(25)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-6, 6), useMathText=True)
    plt.tight_layout()
    plt.savefig(file_dir, bbox_inches="tight");
    plt.close()
    return


def outlier_to_string(outlier_list):
    outlier_list = [str(a) for a in outlier_list]
    # returns comma separated string of the outlier file names
    if len(outlier_list) > 0:
        string = outlier_list[0]
        newline = 1
        for file in outlier_list[1:]:
            string = string + ", "
            if newline % 10 == 0:
                string += "\n"
            string += file
            newline += 1
        return (string)
    else:
        return outlier_list


def biplot_new(principalDf, X_df, pcnum1, pcnum2, title, plot_dir):
    """Creates biplot of 2 user selected principal components and selected scans from the spectral file

    arguments - 8 items

    1. principalDf: the data frame from which PCs are selected
    2. X_df: spectra data as an excel file
    3. pcnum1: first user-selected principal component to plot
    4. pcnum2: second user-selected principal component to plot
    5. title: title of spectra file; agora object attribute
    6. plot_dir: directory where biplot is saved; agora object attribute
    7. pc_spec_dir1: directory where scan for pcnum1 is saved; agora object attribute
    8. pc_spec_dir2: directory where scan for pcnum2 is saved; agora object attribute

    returned by function - 0
    function returns null
    """
    col1 = principalDf.columns[pcnum1 - 1]
    col2 = principalDf.columns[pcnum2 - 1]
    # set up the dataframe for the selected prinicpal components
    temp_df = pd.concat([principalDf[col1], principalDf[col2]], axis=1)
    # add the distances column and run best_fit_distribution()
    temp_df = prep.calculate_distances(temp_df, "distance", col1, col2)
    best_fit_name, best_fit_params, best_fit_int = best_fit_distribution(temp_df["distance"], 200)
    # add the outliers column
    temp_df = prep.create_outlier_group(temp_df, best_fit_int[1], "possible outlier",
                                        "standard", "distance", "group")

    # set the index of the df to the index of X_df
    temp_df.set_index(X_df.index, inplace=True)

    # sort the df for ranking ouf outliers by distance
    temp_df.sort_values(by="distance", ascending=False, inplace=True)

    # set up for the scatter plot
    x_val_title = temp_df.columns[0]
    y_val_title = temp_df.columns[1]

    out_condition = temp_df["group"] == "possible outlier"
    stnd_condition = temp_df["group"] == "standard"

    x_pca = np.array(temp_df[x_val_title])
    y_pca = np.array(temp_df[y_val_title])

    out_all = temp_df[out_condition]["distance"]
    out_x = x_pca[out_condition]
    out_y = y_pca[out_condition]
    stnd_x = x_pca[stnd_condition]
    stnd_y = y_pca[stnd_condition]

    fig1, ax1 = plt.subplots(figsize=fig_size, dpi=400)

    colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(out_x) + 1))
    ax1.scatter(stnd_x, stnd_y, s=3, marker="s", color=colors[0])

    for i in range(len(out_all)):
        ax1.scatter(out_x[i], out_y[i], marker="d", color=colors[i + 1], label=out_all.index[i])

    # get handles and labels for the legend
    handles, labels = ax1.get_legend_handles_labels()
    # check the length of the handles/labels list before assigning to the legend
    elem_handles = 0
    if len(handles) < 11:
        elem_handles = len(handles)
    else:
        elem_handles = 10

    ax1.legend(handles[0:elem_handles], np.arange(0, elem_handles), loc='upper center', bbox_to_anchor=(0.5, -0.15),
               title='Top 10 Outliers', fancybox=True, shadow=True, ncol=5, fontsize=10, title_fontsize=12)

    radius = best_fit_int[1]
    circle = plt.Circle((0, 0), radius, color='gray', fill=False)

    rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': font_desc_stat_sec, 'weight': 'light'})

    ax1.add_artist(circle)

    # rename "principal component n" to "PCn"
    dict_col_titles = {'principal component 1': 'PC1', 'principal component 2': 'PC2',
                       'principal component 3': 'PC3', 'principal component 4': 'PC4',
                       'principal component 5': 'PC5', 'principal component 6': 'PC6'}
    x_axis_title = ""
    y_axis_title = ""
    for key in dict_col_titles:
        if x_val_title == key:
            x_axis_title = dict_col_titles[key]
        if y_val_title == key:
            y_axis_title = dict_col_titles[key]

    ax1.tick_params(labelsize=font_desc_stat_sec)
    ax1.set_title(title, fontsize=font_desc_stat_sec)
    ax1.set_xlabel(x_axis_title, fontsize=font_desc_stat_sec)
    ax1.set_ylabel(y_axis_title, fontsize=font_desc_stat_sec)

    ##save plot
    biplot_dir = "{}/{}".format(plot_dir, title.replace(' ', '_') + '_outliers_biplot.jpg')
    fig1.savefig(biplot_dir, bbox_inches="tight")

    plt.close()


def plot_PCscans(principalDf, X_df, pcnum1, pcnum2, title, pc_spec_dir1, pc_spec_dir2):
    sel_X_df = X_df.copy()
    # reset the index to accomodate use of df.iloc below
    sel_X_df.reset_index(drop=True, inplace=True)

    col1 = principalDf.columns[pcnum1 - 1]
    col2 = principalDf.columns[pcnum2 - 1]
    select_df = pd.concat([principalDf[col1], principalDf[col2]], axis=1)
    # reset the index to accomodate use of df.iloc below
    select_df.reset_index(drop=True, inplace=True)

    # add the distances column and run best_fit_distribution()
    select_df = prep.calculate_distances(select_df, "distance", col1, col2)
    best_fit_name, best_fit_params, best_fit_int = best_fit_distribution(select_df["distance"], 200)
    # add the outliers column
    select_df = prep.create_outlier_group(select_df, best_fit_int[1], "possible outlier",
                                          "standard", "distance", "group")

    x_val_title = select_df.columns[0]
    y_val_title = select_df.columns[1]

    x_col_for_stats = select_df[x_val_title]
    y_col_for_stats = select_df[y_val_title]

    x_pc_min = select_df[x_col_for_stats == x_col_for_stats.min()].index[0]
    x_pc_max = select_df[x_col_for_stats == x_col_for_stats.max()].index[0]
    x_pc_median = select_df[x_col_for_stats == x_col_for_stats.quantile(q=0.50, interpolation='nearest')].index[0]
    x_pc_25th_pct = select_df[x_col_for_stats == x_col_for_stats.quantile(q=0.25, interpolation='nearest')].index[0]
    x_pc_75th_pct = select_df[x_col_for_stats == x_col_for_stats.quantile(q=0.75, interpolation='nearest')].index[0]

    y_pc_min = select_df[y_col_for_stats == y_col_for_stats.min()].index[0]
    y_pc_max = select_df[y_col_for_stats == y_col_for_stats.max()].index[0]
    y_pc_median = select_df[y_col_for_stats == y_col_for_stats.quantile(q=0.50, interpolation='nearest')].index[0]
    y_pc_25th_pct = select_df[y_col_for_stats == y_col_for_stats.quantile(q=0.25, interpolation='nearest')].index[0]
    y_pc_75th_pct = select_df[y_col_for_stats == y_col_for_stats.quantile(q=0.75, interpolation='nearest')].index[0]

    def scan_plotter(ax, df, min_val, max_val, med_val, val_25, val_75):
        ax.plot(df.iloc[min_val], linestyle=':', color='blue', linewidth=0.35, label='minimum')
        ax.plot(df.iloc[max_val], linestyle='--', color='magenta', linewidth=0.35, label='maximum')
        ax.plot(df.iloc[med_val], linestyle='-.', color='grey', linewidth=0.35, label='median')
        ax.plot(df.iloc[val_25], linestyle='-', color='black', linewidth=0.35, label='25th%')
        ax.plot(df.iloc[val_75], linestyle='--', color='cyan', linewidth=0.35, label='75th%')

    fig1, ax1 = plt.subplots(figsize=fig_size, dpi=400)
    fig2, ax2 = plt.subplots(figsize=fig_size, dpi=400)

    scan_plotter(ax1, sel_X_df, x_pc_min, x_pc_max, x_pc_median, x_pc_25th_pct, x_pc_75th_pct)
    scan_plotter(ax2, sel_X_df, y_pc_min, y_pc_max, y_pc_median, y_pc_25th_pct, y_pc_75th_pct)

    rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': font, 'weight': 'light'})

    # rename "principal component n" to "PCn"
    dict_col_titles = {'principal component 1': 'PC1', 'principal component 2': 'PC2',
                       'principal component 3': 'PC3', 'principal component 4': 'PC4',
                       'principal component 5': 'PC5', 'principal component 6': 'PC6'}
    x_axis_title = ""
    y_axis_title = ""
    for key in dict_col_titles:
        if x_val_title == key:
            x_axis_title = dict_col_titles[key]
        if y_val_title == key:
            y_axis_title = dict_col_titles[key]

    x_plot_title = title + ": " + x_axis_title
    y_plot_title = title + ": " + y_axis_title

    ax1.legend(loc='upper right', fontsize=14)
    ax1.tick_params(labelsize=font)
    ax1.set_title(x_plot_title, fontsize=font)
    ax1.set_xlabel('wavelength (nm)', fontsize=font)
    ax1.set_ylabel('intensity', fontsize=font)

    ax2.legend(loc='upper right', fontsize=14)
    ax2.tick_params(labelsize=font)
    ax2.set_title(y_plot_title, fontsize=font)
    ax2.set_xlabel('wavelength (nm)', fontsize=font)
    ax2.set_ylabel('intensity', fontsize=font)

    ##save plots
    pc_spec_dir1 = "{}/{}".format(pc_spec_dir1, title.replace(' ', '_') + '_outliers_pcN1.jpg')
    fig1.savefig(pc_spec_dir1, bbox_inches="tight")

    pc_spec_dir2 = "{}/{}".format(pc_spec_dir2, title.replace(' ', '_') + '_outliers_pcN2.jpg')
    fig2.savefig(pc_spec_dir2, bbox_inches="tight")

    plt.close()


def plot_hist_new(histogram_dataframe, h_title, hist_dir, best_fit_int, distance_col_title):
    """Plots a histogram of the distance column of the dataframe argument

    arguments - 5 items

    1. histogram_dataframe: dataframe containing distance column for histogram
    2. h_title: title for plot; agora object attribute
    3. plot_dir: directory where histogram will be saved; agora attribute
    4. best_fit_int: p-value for the right side of the 95% confidence interval
    5. distance_col_title: used for selection of the distance column in histogram_dataframe

    returned by function - 0
    function returns null

    """

    fig, ax = plt.subplots(figsize=fig_size, dpi=400)

    rwidth = 1.0
    edgecolor = 'k'

    n, bins, patches = ax.hist(histogram_dataframe[distance_col_title], bins='auto', edgecolor=edgecolor, rwidth=rwidth)

    for i in range(len(patches)):
        if bins[i] < best_fit_int:
            patches[i].set_facecolor('b')
        else:
            patches[i].set_facecolor('r')

    xlim_left = histogram_dataframe[distance_col_title].min() - 5
    xlim_right = histogram_dataframe[distance_col_title].max() + 5
    ax.xaxis.set_ticks(np.arange(xlim_left, xlim_right), 5)
    plt.minorticks_off()

    rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': font, 'weight': 'light'})

    ax.set_xlabel('Euclidean Distance from Centroid', fontsize=font_desc_stat_sec)
    ax.set_ylabel('Total Count', fontsize=font_desc_stat_sec)
    ax.set_title('Distribution of Distances from Centroid: ' + h_title, fontsize=font_desc_stat_sec)

    # The import was required here; red_patch and blue_patch are 'Proxy artists' for the legend
    import matplotlib.patches as patches
    red_patch = patches.Patch(color='red', label='Possible outliers')
    blue_patch = patches.Patch(color='blue', label='Standard')
    ax.legend(handles=[red_patch, blue_patch])
    ax.grid(axis='y', alpha=0.25)

    plt.savefig(hist_dir, bbox_inches="tight")

    plt.close()


def ga_scores(scores, title, file_dir):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.scatter(np.arange(len(scores)), scores, c='blue', edgecolors='k')

    plt.xlabel('Generations')
    plt.ylabel('Fitness CV value')
    title = title;
    plt.title(title)

    rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': font, 'weight': 'light'})

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
    ):  # ax.get_xticklabels() + ax.get_yticklabels()
        item.set_fontsize(font)

    for item in ax.get_xticklabels():
        item.set_fontsize(font)
    for item in ax.get_yticklabels():
        item.set_fontsize(font)

    plt.tight_layout()

    fig.savefig(file_dir, bbox_inches="tight");
    plt.close()


def plot_val_curve(train_scores, test_scores, param_range,param_name, metric,ml_method,fig_dir):

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # find the parameter within one standard error of the best CV test score: looking for a less complex but as accurate model
    ses = st.sem(test_scores, axis=1)
    ind = test_scores_mean.argmax()
    score = test_scores_mean[ind]
    diff = abs(test_scores_mean - score)
    arg_param = np.argmax(diff < ses[ind])

    fig, ax =plt.subplots(figsize=(9, 5), dpi=300)
    font = 18
    plt.title("Cross Validation Curve with "+ ml_method, fontsize=font)
    plt.xlabel('Model Complexity Parameter: '+ param_name, fontsize=font)
    plt.ylabel(metric, fontsize=font)
    lw = 2
    if param_name=='n_components':
        ax.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", linestyle="--", lw=lw)


        ax.errorbar(param_range, test_scores_mean, yerr=ses, label="CV Test score with One SE Bar",
                     color="navy", lw=lw, linestyle='-')
        ax.plot(param_range[arg_param], test_scores_mean[arg_param], '*', ms=10, mfc='blue', label='Selected Parameter')
        ax.fill_between(param_range, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.3,
                        color="darkorange", lw=lw)
        ax.fill_between(param_range, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.3,
                        color="navy", lw=lw)

    else:

        ax.semilogx(param_range, train_scores_mean, label="Training score",
                color="darkorange", linestyle="--", lw=lw)
        ax.errorbar(param_range, test_scores_mean, yerr=ses, label="CV Test score with One SE Bar",
                    color="navy", lw=lw, linestyle='-')
        ax.semilogx(param_range[arg_param], test_scores_mean[arg_param], '*', ms=10, mfc='blue', label='Selected Parameter')
        ax.fill_between(param_range, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.3,
                        color="darkorange", lw=lw)
        ax.fill_between(param_range, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.3,
                        color="navy", lw=lw)
        ax.set_xscale('log', basex=2)


    plt.legend(loc="best", fontsize=font)

    rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': font, 'weight': 'light'})


    plt.tight_layout()

    plt.savefig(fig_dir, bbox_inches="tight");

    plt.close()

def plot_ncomps_vs_exp_var(n_comps, var_exp, fig_dir):

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    font = 18
    plt.xlabel('Number of Principal Components', fontsize=font)
    plt.ylabel('Variance Explained (%)',fontsize=font)
    comp_list=list(range(n_comps))
    for n in comp_list:
        comp_list[n]=comp_list[n]+1
    total_var = 0
    tot_var_list = []
    min_ncomp_list = []
    for i in range(n_comps):
        total_var += var_exp[i]
        if total_var >= .99:
            min_ncomp_list.append(i)
        tot_var_list.append(total_var * 100) #to be listed as a percentage
        if len(min_ncomp_list) >= 1:
            min_ncomp = str(min_ncomp_list[0]+1)
            plt.title('Minimum Number of PCs to Explain 99% of Variance: '+min_ncomp, fontsize=font)
        else:
            n_comps = str(n_comps)
            plt.title('Minimum Number of PCs to Explain 99% Variance > '+n_comps, fontsize=font)
    plt.bar(comp_list, tot_var_list, color="blue", edgecolor="black", width=0.6)
    rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': font, 'weight': 'light'})
    plt.tight_layout()
    plt.savefig(fig_dir, bbox_inches="tight")
    plt.close()

    return