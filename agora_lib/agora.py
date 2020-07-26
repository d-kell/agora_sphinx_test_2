import os
import pickle
import sys
import warnings
import pandas as pd
from xlrd import XLRDError
import numpy as np
warnings.filterwarnings("ignore")
import datetime
import time
import random
import shutil
import fire
from agora_lib import agora_pipeline, plotting, ml, pdf_gen, prep, simca
import logging
import joblib
from google.cloud import storage
from zipfile import ZipFile
import matplotlib

matplotlib.use('Agg')
from sklearn.metrics import r2_score

font_model_sec = 22
font_desc_stat_sec = 22


class Agora:
    attrs = None
    all_train_outlrs = None
    all_val_outlrs = None
    req_df = None
    ranked_train_outlrs = None
    ranked_val_outlrs = None
    val_set_given = None
    stop = False

    def __init__(self, file_dir, file_dir_test, plot_dir, remove_out_Tr, remove_out_Val, rep_corr,
                 average, read_err, **kwargs):
        self.niter = None
        self.popSize = None
        self.cutoff = None
        self.gen_pdf = True
        # get a list of all predefined values directly from __dict__
        allowed_keys = list(self.__dict__.keys())
        self.__dict__.update((key, value) for key, value in kwargs.items()
                             if key in allowed_keys)

        self.file_dir, self.file_dir_test, self.plot_dir = file_dir, file_dir_test, plot_dir
        self.read_err = read_err
        self.check_dir = prep.make_subf(plot_dir, 'Data_Check_Figures')

        self.remove_out_Tr = remove_out_Tr
        self.remove_out_Val = remove_out_Val
        self.rep_corr = rep_corr
        # All raw spec directory
        self.raw_spec_dir = prep.make_subf(self.plot_dir, 'Raw_Spectra_Figures')
        if len(file_dir_test) > 0:
            self.val_set_given = True
            self.raw_val_spec = "{}/{}".format(self.raw_spec_dir, 'Validation_raw_spec.jpg')
            self.raw_tr_spec = "{}/{}".format(self.raw_spec_dir, 'Training_raw_spec.jpg')


        else:
            self.raw_tr_spec = "{}/{}".format(self.raw_spec_dir, 'raw_spec.jpg')
            self.val_set_given = False

        self.clean_tr_spec = "{}/{}".format(self.raw_spec_dir, 'Training_clean_spectra.jpg')
        if self.remove_out_Val:
            self.clean_val_spec = "{}/{}".format(self.raw_spec_dir, 'Validation_clean_spectra.jpg')

        ######## outlier directories

        self.outl_dir = prep.make_subf(self.plot_dir, 'Outlier_Figures')
        self.outl_tr_dir = prep.make_subf(self.outl_dir, 'Outliers_Training')
        if self.val_set_given:
            self.outl_val_dir = prep.make_subf(self.outl_dir, 'Outliers_Validation')
        self.average = average
        self.pkl_dir = prep.make_subf(self.plot_dir, 'Model_objects')  # create a sub folder to output the results
        self.pred_dir = prep.make_subf(self.plot_dir, 'Predictions')
        self.outl_csv = prep.make_subf(self.plot_dir, 'Outlier_CVSs')
        self.all_model_dir = prep.make_subf(self.plot_dir, 'Model_Figures')
        self.all_spec_dir = prep.make_subf(self.plot_dir, 'Processed_Spectra_Figures')

        self.simca = None
        self.universal = None
        self.start = datetime.datetime.now()
        self.rep_ids, self.rep_ids_val = None, None

    def load_data(self, excel_dir, raw_spec_dir, outl_dir, title, remove_out=True, load_req=True, read_err=False):
        ##Load Data From Excel Workbook##
        rep_ids = []
        xls = pd.ExcelFile(excel_dir)

        if load_req:
            try:
                self.req_df = xls.parse(sheet_name='Request')
                if read_err:
                    accepted_errs = xls.parse(sheet_name='Accepted Error')
                    self.accepted_errs = accepted_errs.set_index(accepted_errs.columns[0])
            except XLRDError:
                print('Check the "Request" or "Accepted Error" sheet name and  content')
                sys.exit(1)

            try:
                self.met_df = xls.parse(sheet_name='MetaData')
            except XLRDError:
                print('Check the "MetaData" sheet name and  content')
                sys.exit(1)
        try:
            Y_df = xls.parse(sheet_name='Reference Data (Y)')  # Reference data on 3d sheet
            X_df = xls.parse(sheet_name='Raw Spectra (X)')  # Raw spectra on 4th sheet
        except XLRDError:
            print('Required reference and spectra sheet names: "Reference Data (Y)" and "Raw Spectra (X)')
            sys.exit(1)
        Y_df = Y_df.set_index(Y_df.columns[0])
        X_df = X_df.set_index(X_df.columns[0])
        X_df = X_df.dropna(axis=0)


        all_outliers, ranked_outliers = [], []
        if 'Replicate Group ID' not in X_df.columns and self.rep_corr:
            logging.error(
                'Replicate correction requested but no Replicate  Group ID columns present! Column name given:' + str(
                    X_df.columns[0]))
            sys.exit(1)
        elif self.rep_corr and 'Replicate Group ID' in X_df.columns :

            if X_df.columns[1] is str:
                X_df.columns[1:] = [float(a) for a in X_df.columns[1:]]

            X_df, Y_df, rep_ids, all_outliers, ranked_outliers = prep.corr_rep(X_df, Y_df, raw_spec_dir, outl_dir,
                                                                               title,
                                                                               remove_out)
        elif not self.rep_corr and 'Replicate Group ID' in X_df.columns:
            if X_df.columns[1] is str:
                X_df.columns[1:] = [float(a) for a in X_df.columns[1:]]
            rep_ids = X_df['Replicate Group ID']
            X_df.drop('Replicate Group ID', axis=1, inplace=True)
            Y_df.drop('Replicate Group ID', axis=1, inplace=True)
            X_df.columns = [float(a) for a in X_df.columns]
            all_outliers, ranked_outliers = prep.save_outlier_plots_new(X_df, outl_dir, title=title)
            plotting.plot_spectra(X_df.columns, X_df.to_numpy(), 'Raw ' + title + ' Spectra',
                                  font_size=font_desc_stat_sec, save_fig=True,
                                  file_dir=raw_spec_dir)
            plotting.plot_spectra(X_df.columns.astype(float), X_df.loc[all_outliers, :].to_numpy(),
                                  title + ' Outliers Spectra ', font_size=font_desc_stat_sec,
                                  file_dir="{}/{}".format(outl_dir, title + '_outliers_spec.jpg'), save_fig=True)

            if remove_out:
                X_df = X_df.drop(all_outliers)
                Y_df = Y_df.drop(all_outliers)

        else:
            X_df.columns = [float(a) for a in X_df.columns]
            all_outliers, ranked_outliers = prep.save_outlier_plots_new(X_df, outl_dir, title=title)

            plotting.plot_spectra(X_df.columns, X_df.to_numpy(), 'Raw ' + title + ' Spectra', colors="spectral", font_size=font_desc_stat_sec,
                                  alloutliers=X_df.loc[all_outliers, :].to_numpy(), rankedoutliers= X_df.loc[ranked_outliers, :].to_numpy(),
                                  save_fig=True, file_dir=raw_spec_dir)
            plotting.plot_spectra(X_df.columns.astype(float), X_df.loc[all_outliers, :].to_numpy(),
                                  title + ' Outliers Spectra ', colors="spectral", font_size=font_desc_stat_sec,
                                  file_dir="{}/{}".format(outl_dir, title + '_outliers_spec.jpg'), save_fig=True)

            if remove_out:
                X_df = X_df.drop(all_outliers)
                Y_df = Y_df.drop(all_outliers)

        if X_df.columns[1] > X_df.columns[2]:  # wavenumbers are in decreasing order
            X_df = X_df.iloc[:, ::-1]

        return X_df, Y_df, rep_ids, all_outliers, ranked_outliers

    def load_dfs(self):

        all_val_outliers, ranked_val_outliers, rep_ids_val = [], [], []

        def Q_hotT2(X, Y, title, outl_dir):
            addnl_out = prep.out_T2_Q(X, Y.iloc[:, 0], outl_dir, title)
            X.loc[addnl_out, :].to_csv(self.outl_csv + '/Q_hotT2_outl_spec.csv')
            Y.loc[addnl_out, :].to_csv(self.outl_csv + '/Q_hotT2_outl_ref.csv')
            return addnl_out

        def get_addnl_outl(X, Y, title, all_outliers, remove_out, outl_dir):

            addnl_out = Q_hotT2(X, Y, title, outl_dir)
            all_outliers += list(addnl_out)
            all_outliers = list(pd.DataFrame(data=all_outliers, columns=['Possible_Outlier']).drop_duplicates().loc[:,
                                'Possible_Outlier'])
            if remove_out:
                for outl in addnl_out:
                    if outl in X.index:
                        X.drop(outl, inplace=True)
                        Y.drop(outl, inplace=True)
            return X, Y, all_outliers

        if self.val_set_given:

            X_df, Y_df, rep_ids, all_outliers, ranked_outliers = self.load_data(self.file_dir, self.raw_tr_spec,
                                                                                self.outl_tr_dir, title='Training',
                                                                                remove_out=self.remove_out_Tr,
                                                                                load_req=True, read_err=self.read_err)

            X_valid, Y_valid, rep_ids_val, all_val_outliers, ranked_val_outliers = self.load_data(self.file_dir_test,
                                                                                                  self.raw_val_spec,
                                                                                                  self.outl_val_dir,
                                                                                                  title='Validation',
                                                                                                  remove_out=self.remove_out_Val,
                                                                                                  load_req=False,
                                                                                                  read_err=False)

            self.q_res_hot_t2_dir = "{}/{}".format(self.outl_val_dir, 'Validation_outliers_biplot_T2_Q.jpg')

            X_valid, Y_valid, all_val_outliers = get_addnl_outl(X_valid, Y_valid, 'Validation', all_val_outliers,
                                                                self.remove_out_Val, self.q_res_hot_t2_dir)
            self.attrs = Y_valid.columns


            if self.rep_corr:

                self.hist_tr_dir = self.outl_tr_dir + '/Training_Before_EMSC_outliers_hist.jpg'
                self.biplot_tr_dir = self.outl_tr_dir + '/Training_Before_EMSC_outliers_biplot.jpg'
                self.exp_var_dir = self.outl_tr_dir + '/Training_Before_EMSC_PCcomp_vs_var_exp.jpg'
                self.outl_tr_spec = self.outl_tr_dir + '/Training_Before_EMSC_outliers_spec.jpg'
                self.tr_pc1_spec = self.outl_tr_dir + '/Training_Before_EMSC_outliers_pcN1.jpg'
                self.tr_pc2_spec = self.outl_tr_dir + '/Training_Before_EMSC_outliers_pcN2.jpg'

                self.hist_tr_dir_emsc = self.outl_tr_dir + '/Training_After_EMSC_outliers_hist.jpg'
                self.biplot_tr_dir_emsc = self.outl_tr_dir + '/Training_After_EMSC_outliers_biplot.jpg'
                self.exp_var_dir = self.outl_tr_dir + '/Training_After_EMSC_PCcomp_vs_var_exp.jpg'
                self.outl_tr_spec_emsc = self.outl_tr_dir + '/Training_After_EMSC_outliers_spec.jpg'
                self.tr_pc1_spec_emsc = self.outl_tr_dir + '/Training_After_EMSC_outliers_pcN1.jpg'
                self.tr_pc2_spec_emsc = self.outl_tr_dir + '/Training_After_EMSC_outliers_pcN2.jpg'

                #### Validation directories

                self.hist_val_dir = self.outl_val_dir + '/Validation_Before_EMSC_outliers_hist.jpg'
                self.biplot_val_dir = self.outl_val_dir + "/Validation_Before_EMSC_outliers_biplot.jpg"
                self.exp_var_dir = self.outl_val_dir + '/Validation_Before_EMSC_PCcomp_vs_var_exp.jpg'
                self.outl_val_spec = self.outl_val_dir + '/Validation_Before_EMSC_outliers_spec.jpg'
                self.val_pc1_spec = self.outl_val_dir + '/Validation_Before_EMSC_outliers_pcN1.jpg'
                self.val_pc2_spec = self.outl_val_dir + '/Validation_Before_EMSC_outliers_pcN2.jpg'

                self.hist_val_dir_emsc = self.outl_val_dir + '/Validation_After_EMSC_outliers_hist.jpg'
                self.biplot_val_dir_emsc = self.outl_val_dir + "/Validation_After_EMSC_outliers_biplot.jpg"
                self.exp_var_dir = self.outl_val_dir + '/Validation_After_EMSC_PCcomp_vs_var_exp.jpg'
                self.outl_val_spec_emsc = self.outl_val_dir + '/Validation_After_EMSC_outliers_spec.jpg'
                self.val_pc1_spec_emsc = self.outl_val_dir + '/Validation_After_EMSC_outliers_pcN1.jpg'
                self.val_pc2_spec_emsc = self.outl_val_dir + '/Validation_After_EMSC_outliers_pcN2.jpg'

            else:
                self.hist_val_dir = self.outl_val_dir + '/Validation_outliers_hist.jpg'
                self.biplot_val_dir = self.outl_val_dir + '/Validation_outliers_biplot.jpg'
                self.exp_var_dir = self.outl_val_dir + '/Validation_PCcomp_vs_var_exp.jpg'
                self.outl_val_spec = self.outl_val_dir + '/Validation_outliers_spec.jpg'
                self.val_pc1_spec = self.outl_val_dir + '/Validation_outliers_pcN1.jpg'
                self.val_pc2_spec = self.outl_val_dir + '/Validation_outliers_pcN2.jpg'

                self.hist_tr_dir = self.outl_tr_dir + '/Training_outliers_hist.jpg'
                self.biplot_tr_dir = self.outl_tr_dir + '/Training_outliers_biplot.jpg'
                self.exp_var_dir = self.outl_tr_dir + '/Training_PCcomp_vs_var_exp.jpg'
                self.outl_tr_spec = self.outl_tr_dir + '/Training_outliers_spec.jpg'
                self.tr_pc1_spec = self.outl_tr_dir + '/Training_outliers_pcN1.jpg'
                self.tr_pc2_spec = self.outl_tr_dir + '/Training_outliers_pcN2.jpg'





        else:
            X_df, Y_df, rep_ids, all_outliers, ranked_outliers = self.load_data(self.file_dir, self.raw_tr_spec,
                                                                                self.outl_tr_dir, title='Training',
                                                                                remove_out=self.remove_out_Tr,
                                                                                load_req=True, read_err=self.read_err)
            self.attrs = Y_df.columns

            if self.rep_corr:

                self.hist_tr_dir = self.outl_tr_dir + '/Training_Before_EMSC_outliers_hist.jpg'
                self.biplot_tr_dir = self.outl_tr_dir + '/Training_Before_EMSC_outliers_biplot.jpg'
                self.exp_var_dir = self.outl_tr_dir + '/Training_Before_EMSC_PCcomp_vs_var_exp.jpg'
                self.outl_tr_spec = self.outl_tr_dir + '/Training_Before_EMSC_outliers_spec.jpg'
                self.tr_pc1_spec = self.outl_tr_dir + '/Training_Before_EMSC_outliers_pcN1.jpg'
                self.tr_pc2_spec = self.outl_tr_dir + '/Training_Before_EMSC_outliers_pcN2.jpg'

                self.hist_tr_dir_emsc = self.outl_tr_dir + '/Training_After_EMSC_outliers_hist.jpg'
                self.biplot_tr_dir_emsc = self.outl_tr_dir + '/Training_After_EMSC_outliers_biplot.jpg'
                self.exp_var_dir = self.outl_tr_dir + '/Training_After_EMSC_PCcomp_vs_var_exp.jpg'
                self.outl_tr_spec_emsc = self.outl_tr_dir + '/Training_After_EMSC_outliers_spec.jpg'
                self.tr_pc1_spec_emsc = self.outl_tr_dir + '/Training_After_EMSC_outliers_pcN1.jpg'
                self.tr_pc2_spec_emsc = self.outl_tr_dir + '/Training_After_EMSC_outliers_pcN2.jpg'


            else:
                self.hist_tr_dir = self.outl_tr_dir + '/Training_outliers_hist.jpg'
                self.biplot_tr_dir = self.outl_tr_dir + '/Training_outliers_biplot.jpg'
                self.exp_var_dir = self.outl_tr_dir + '/Training_PCcomp_vs_var_exp.jpg'
                self.outl_tr_spec = self.outl_tr_dir + '/Training_outliers_spec.jpg'
                self.tr_pc1_spec = self.outl_tr_dir + '/Training_outliers_pcN1.jpg'
                self.tr_pc2_spec = self.outl_tr_dir + '/Training_outliers_pcN2.jpg'
            X_valid = 0
            Y_valid = 0

        self.all_train_outlrs = all_outliers
        self.all_val_outlrs = all_val_outliers
        self.ranked_train_outlrs = ranked_outliers
        self.ranked_val_outlrs = ranked_val_outliers
        self.rep_ids, self.rep_ids_val = rep_ids, rep_ids_val

        logging.info("Loaded the data and performed an outlier detection: {}".format(self.plot_dir))

        if len(all_outliers) > 0:  # save outliers as csv files
            pd.DataFrame(data=all_outliers, columns=['Outliers_Training_ID']).drop_duplicates().to_csv(
                self.outl_csv + '/Outliers_Training.csv')

        if self.val_set_given and len(all_val_outliers) > 0:
            pd.DataFrame(data=all_val_outliers, columns=['Outliers_Validation_ID']).to_csv(
                self.outl_csv + '/Outliers_Validation.csv')

        return X_df, Y_df, X_valid, Y_valid

    def merge_data(self, X_df, Y_df, X_valid, Y_valid):
        wl = X_df.columns
        data_dir = prep.make_subf(self.plot_dir, 'Merged_Data_csvs')  # create a sub folder to output the results
        merged_df_val = []

        if self.val_set_given:
            merged_df_val = Y_valid.merge(X_valid, on=Y_valid.index.name, how='inner')
            merged_df_val.to_csv(data_dir + '/merged_df_validation.csv')
            if self.remove_out_Val:
                plotting.plot_spectra(wl, X_valid.loc[:, wl].to_numpy(), title='Clean Validation Spectra',
                                      colors="binary", save_fig=True, font_size=font_desc_stat_sec,
                                      file_dir=self.clean_val_spec)
            title = 'Clean Training Spectra'


        else:
            title = 'Spectra before splitting into train and test'
            merged_df_val = ''

        plotting.plot_spectra(wl, X_df.to_numpy(), title,  font_size=font_desc_stat_sec,colors="binary", save_fig=True, file_dir=self.clean_tr_spec)

        merged_df = Y_df.merge(X_df, on=Y_df.index.name, how='inner')
        merged_df.to_csv(data_dir + '/merged_df_training.csv')
        self.wl = merged_df.columns[len(Y_df.columns):]

        return merged_df, merged_df_val

    def merge_by_attr(self, merged_df, attr):

        Y = merged_df.loc[:, attr]
        wl = self.wl
        if 'Osmo' not in attr and 'Sucrose' not in attr and 'Met' not in attr and 'PS20' not in attr:  # use only 400-1850 wavenumbers if it's not Osmo
            if 440.0 in self.wl and 1851.0 in self.wl:
                wl = self.wl[list(self.wl).index(440.0):list(self.wl).index(1851.0)]
            elif '440.0' in self.wl and '1851.0' in self.wl:
                wl = self.wl[list(self.wl).index('440.0'):list(self.wl).index('1851.0')]

        df = pd.concat([Y, merged_df.loc[:, wl]], axis=1).dropna()

        Y = df[attr]
        spectra = df.loc[:, wl]

        return spectra, Y, wl, df.index

    def prep_matrices(self, attr):
        spectra, Y, wl_attr, pred_idx = self.merge_by_attr(self.merged_df, attr)
        score = 1
        # score, param=self.tr_data_check(x_train=spectra, y_train=Y, plot_dir=self.check_dir+'/'+attr.replace('/','')+'_tr_data_check.jpg')

        if self.val_set_given:

            x_valid, y_valid, wl_val, pred_idx = self.merge_by_attr(self.merged_df_val, attr)

            if self.rep_corr:

                pd.DataFrame(Y).merge(self.rep_ids, on=Y.index.name).loc[:, 'Replicate Group ID'].to_csv(
                    self.pred_dir + '/Training_Replicate_Group_ID_File_IDs.csv')

                pd.DataFrame(y_valid).merge(self.rep_ids_val, on=y_valid.index.name).loc[:,
                'Replicate Group ID'].to_csv(
                    self.pred_dir + '/Validation_Replicate_Group_ID_File_IDs.csv')

                if self.average:
                    spectra = spectra.merge(self.rep_ids, on=spectra.index.name).groupby('Replicate Group ID').mean()
                    Y = pd.DataFrame(Y).merge(self.rep_ids, on=Y.index.name).groupby('Replicate Group ID').mean()

                    x_valid = x_valid.merge(self.rep_ids_val, on=x_valid.index.name).groupby(
                        'Replicate Group ID').mean()
                    y_valid = pd.DataFrame(y_valid).merge(self.rep_ids_val, on=y_valid.index.name).groupby(
                        'Replicate Group ID').mean()
                else:
                    print('Did  not average')
            pred_idx = y_valid.index

            x_valid = x_valid.to_numpy()
            y_valid = y_valid.to_numpy()

        else:

            if self.rep_corr:
                pd.DataFrame(Y).merge(self.rep_ids, on=Y.index.name).loc[:, 'Replicate Group ID'].to_csv(
                    self.pred_dir + '/Replicate_Group_ID_File_IDs.csv')
                if self.average:
                    spectra = spectra.merge(self.rep_ids, on=spectra.index.name).groupby('Replicate Group ID').mean()
                    Y = pd.DataFrame(Y).merge(self.rep_ids, on=Y.index.name).groupby('Replicate Group ID').mean()

            x_valid = np.zeros([1])
            y_valid = np.zeros([1])
            pred_idx = Y.index

        self.pred_idx = pred_idx

        # ###SNR indices
        # signal_wavenum = 850
        # noise_wavenum = 1800
        # signal_index = spectra.columns.get_loc(signal_wavenum)
        # noise_index = spectra.columns.get_loc(noise_wavenum)
        # self.signal_id = int(spectra.iloc[0, signal_index - 5:signal_index + 5].argmax())
        # self.signal_col = pd.DataFrame.to_numpy(spectra[self.signal_id])
        # self.noise_id = int(spectra.iloc[0, noise_index - 5:noise_index + 5].argmax())
        # self.noise_col = pd.DataFrame.to_numpy(spectra[self.noise_id])

        return spectra.to_numpy(), Y.to_numpy().flatten(), x_valid, y_valid.flatten(), wl_attr, score < 0

    def prep_data(self):
        X_df, Y_df, X_valid, Y_valid = self.load_dfs()
        self.merged_df, self.merged_df_val = self.merge_data(X_df, Y_df, X_valid, Y_valid)

    def save_object(self, obj, name):
        output = open(self.pkl_dir + '/' + name, 'wb')
        pickle.dump(obj, output)  # dump GA pickled object
        output.close()

    def save_agora(self, name):
        output = open(self.pkl_dir + '/' + name, 'wb')
        pickle.dump(self, output)  # dump GA pickled object
        output.close()

    def tr_data_check(self, x_train, y_train, plot_dir):
        [score, param] = ml.pls(x_train, y_train, metric='r2', cv=5, plot_components=True, plot_dir=plot_dir)
        return score, param

    def val_data_check(self, x_train, y_train, x_test, y_test, param):
        y_pred, pls = ml.pls_final_result(x_train, y_train, x_test, param)
        score = r2_score(y_test, y_pred)

        return score, y_pred

    def create_model(self, attr, model_name):
        spectra, Y, x_valid, y_valid, wl_attr, stop_flag = self.prep_matrices(attr)
        if not stop_flag:

            attr_model_dir = prep.make_subf(self.all_model_dir, attr.replace('/', ''))
            attr_spec_dir = prep.make_subf(self.all_spec_dir, attr.replace('/', ''))
            if self.average:  # no need for groupped shuffle split
                rep_ids_val, rep_ids = [], []
            else:
                rep_ids_val, rep_ids = self.rep_ids_val, self.rep_ids

            if self.read_err:
                errs = self.accepted_errs
                accepted_err = errs.loc[attr, 'Accepted Relative Error']
            else:
                accepted_err = 0.15
            if model_name == 'simca':

                self.model = simca.SIMCA(model_name=model_name, attr=attr, val_set_given=self.val_set_given, plot_dir=self.plot_dir,
                                         wl=wl_attr,
                                         rep_ids=rep_ids,
                                         rep_ids_val=rep_ids_val,
                                         pred_dir=self.pred_dir, pkl_dir=self.pkl_dir, spec_dir=attr_spec_dir,
                                         model_dir=attr_model_dir, accepted_err=accepted_err, niter=self.niter,
                                         popSize=self.popSize, cutoff=self.cutoff)
                                         # signal_col=self.signal_col, noise_col=self.noise_col)
            else:

                self.model = agora_pipeline.Agora_pipeline(model_name=model_name, attr=attr, val_set_given=self.val_set_given, plot_dir=self.plot_dir, wl=wl_attr,
                                                           rep_ids=rep_ids,
                                                           rep_ids_val=rep_ids_val,
                                                           pred_dir=self.pred_dir, pkl_dir=self.pkl_dir, spec_dir=attr_spec_dir,
                                                           model_dir=attr_model_dir, accepted_err=accepted_err, niter=self.niter,
                                                           popSize=self.popSize, cutoff=self.cutoff)
                                                           # signal_col=self.signal_col, noise_col=self.noise_col)
            if spectra.shape[0] > 500:
                self.cv = 10
            else:
                self.cv = 5
            self.model.split_data(spectra, Y, x_valid, y_valid, self.pred_idx)
        else:
            logging.warning("Negative Prediction R2, check your data for: {}!".format(attr))
        return stop_flag

    def get_pipeline(self, attr, model_name):
        stop_flag = self.create_model(attr=attr, model_name=model_name)
        if not stop_flag:
            pipeline=self.model.get_results()
        else:
            if len(self.attrs) > 1:
                self.gen_pdf = False
                pipeline=None

        return pipeline


    def get_pipelines(self, model_name):
        all_models = {}
        for i in range(len(self.attrs)):

            pipeline = self.get_pipeline(attr=self.attrs[i], model_name=model_name)

            all_models[self.attrs[i]] = pipeline
            
        self.save_object(all_models, 'All_models_' + model_name + '.pkl')
        setattr(self, model_name, all_models)

    def email_pdf(self):

        start = time.time()

        if self.gen_pdf:
            pdf_gen.make_pdf(self)
            stop = time.time()
            print('Your report is generated in %5.3f seconds' % round(stop - start, 3))
        # contents = "<h1>https://drive.google.com/open?id=1jQNRpi_j3F96Nv5Cib6cGRyGlvvkYDZs</h1> \n " \
        #            "Agora run was done in " + str((stop0 - start0) / 60) + " minutes. Please find the auto-report and supporting plots in the folder: " + plot_dir
        # ######Specify sender/reciever email addresses##########
        #
        # yagmail.SMTP(sender).send(receiver, 'Your Report is Ready!', contents)


def upload_zip_to_gs(plot_dir, gs_bucket, run_tag):
    print(os.listdir())
    # create a zipfile object
    print('making a zip file from ' + plot_dir)
    zipName = plot_dir.split('/')[-1] + '.zip'
    zip_dir = '../' + zipName
    with ZipFile(zip_dir, 'w') as zipObj:
        # Iterate over all the files in directory
        for fpath in os.listdir():
            if os.path.isdir(fpath):
                for folderName, subfolders, filenames in os.walk(fpath):
                    for filename in filenames:
                        filePath = os.path.join(folderName, filename)

                        zipObj.write(filePath)
            else:
                if fpath != 'credentials.json':
                    zipObj.write(fpath)

    zipObj.close()
    print('opening a client for gs bucket')
    client = storage.Client()
    bucket = client.get_bucket(gs_bucket)
    output = os.path.join(run_tag, zipName)
    blob = bucket.blob(output)
    blob.upload_from_filename(filename=zip_dir)
    print('the zip file is copied to the gs bucket!')


def main_body(file_dir, file_dir_test, plot_dir, remove_out_Tr, remove_out_Val, rep_corr, average, read_err, **kwargs):
    agora_obj = Agora(file_dir=file_dir, file_dir_test=file_dir_test, plot_dir=plot_dir, remove_out_Tr=remove_out_Tr,
                      remove_out_Val=remove_out_Val, rep_corr=rep_corr, average=average, read_err=read_err, **kwargs)
    agora_obj.prep_data()

    agora_obj.get_pipelines('simca')
    agora_obj.get_pipelines('universal')
    agora_obj.end = datetime.datetime.now()

    agora_obj.save_agora('agora_obj.pkl')

    agora_obj.email_pdf()


def prep_dirs(plot_dir, run_tag):
    plot_dir = prep.make_subf(plot_dir,
                              run_tag + str(datetime.datetime.now()).replace(":","_"))  # create a sub folder to output the results
    header = os.path.join(plot_dir, 'header')
    shutil.copytree('./header', header)  # copy header figure for the pdf
    os.chdir(plot_dir)  # change working directory to the root folder
    return plot_dir


def gcp_train(job_dir, file_dir, gs_bucket, niter=5, popSize=24, cutoff=6, run_tag='test', file_dir_test='',
              remove_out_Tr=False, remove_out_Val=False, rep_corr=True, average=True, read_err=False):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'credentials.json'

    plot_dir = prep_dirs(plot_dir=os.getcwd(), run_tag=run_tag)
    shutil.copyfile('credentials.json', os.path.join(plot_dir, 'credentials.json'))  # copy json credentials file

    # load all the dataframes and perform outlier analysis
    logging.info("Loading data: {}".format(file_dir))
    univ_models = main_body(file_dir, file_dir_test, plot_dir, niter, popSize, cutoff, remove_out_Tr, remove_out_Val,
                            rep_corr,
                            average, read_err)
    upload_zip_to_gs(plot_dir, gs_bucket, run_tag)

    model_filename = run_tag + '_model.joblib'
    joblib.dump(value=univ_models, filename=model_filename)

    client = storage.Client()
    bucket = client.get_bucket(job_dir.split('/')[-2])
    blob = bucket.blob(job_dir.split('/')[-1] + '/' + model_filename)
    blob.upload_from_filename(filename=model_filename)

    gcs_model_path = "{}/{}".format(job_dir, model_filename)

    logging.info("Saved model in: {}".format(gcs_model_path))


def web_app_run(file_dir, plot_dir, file_dir_test='', run_tag='test',
              remove_out_Tr=True, remove_out_Val=False, rep_corr=False, average=True, read_err=False, niter=0, popSize=12, cutoff=4):
    plot_dir = prep_dirs(plot_dir, run_tag)  # create a subfolder labeled with run tag + datetime

    def get_pipelines(model_name):
        agora_obj = Agora(file_dir=file_dir, file_dir_test=file_dir_test, plot_dir=plot_dir,
                          remove_out_Tr=remove_out_Tr, remove_out_Val=remove_out_Val, rep_corr=rep_corr,
                          average=average, read_err=read_err, niter=niter, popSize=popSize, cutoff=cutoff)
        agora_obj.prep_data()
        agora_obj.get_pipelines(model_name)

        return agora_obj

    agora_simca = get_pipelines(model_name='simca')
    agora_univ = get_pipelines(model_name='universal')

    agora_univ.simca = agora_simca.simca  # add simca models to agora object with universal models
    agora_univ.end = datetime.datetime.now()
    agora_univ.save_agora('agora_obj.pkl')  # both simca and universal models object

    agora_univ.email_pdf()
    logging.info("Saved results in: {}".format(plot_dir))


def local_run(file_dir, plot_dir, file_dir_test='', run_tag='test',
              remove_out_Tr=True, remove_out_Val=False, rep_corr=False, average=True, read_err=False, **kwargs):
    plot_dir = prep_dirs(plot_dir, run_tag)  # create a subfolder labeled with run tag + datetime
    univ_models = main_body(file_dir, file_dir_test, plot_dir='./', remove_out_Tr=remove_out_Tr,
                            remove_out_Val=remove_out_Val, rep_corr=rep_corr,
                            average=average, read_err=read_err, **kwargs)

    print('Local analysis is done!')
    logging.info("Saved results in: {}".format(plot_dir))


if __name__ == '__main__':
    random.seed(1234)

    logging.basicConfig(level=logging.INFO)

    fire.Fire({
        'local_run': local_run,
        'gcp_train': gcp_train,
        'web_app_run': web_app_run#(plot_dir='.',
        #                            file_dir='../agora_lib/data/scaled_ph_data/Agora_Submission_Test_Short.xlsx',
        #                            file_dir_test='../agora_lib/data/scaled_ph_data/Agora_Submission_Test_Short.xlsx',
        #                            niter=1, popSize=2, cutoff=2)
    })

