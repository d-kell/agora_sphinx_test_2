import datetime
import os

from fpdf import FPDF
from datetime import date
from agora_lib import plotting

from PIL import Image

def image_rescale(ref_image_height, input_image_dir):
    input_image = Image.open(input_image_dir)

    asp_ratio = input_image.size[0] / input_image.size[1]  # image width divided by image height to get aspect ratio

    h_final = int(ref_image_height)  # Image height --> equivalent to the reference image
    w_final = float(asp_ratio * h_final)  # adjust image width based on the aspect ratio

    return w_final, h_final  # Return rescaled image dimensions: width_final and height_final


def write_requestor_details(req_df):
    """ Writes out details of the requestor """

    header_image = os.getenv('HEADER_IMG_PATH', 'header/Report_Noodles.png')
    header_text = str(req_df.iloc[0, 1]) + "_" + str(datetime.datetime.now())

    class PDF(FPDF):
        ###FPDF measures A4 paper, units in millimeters MDSR left/right margins = 0.79" =~ 20 mm
        def header(self):
            # Logo
            self.image(header_image, 10, 8, 33)
            # Times bold 15
            self.set_font('Times', 'B', 15)
            # Move to the right
            self.cell(80)
            # Title
            self.cell(30, 10, header_text, 0, 0, 'C')
            # Line break
            # self.ln(20)

        # Page footer
        def footer(self):
            # Position at 1.5 cm from bottom
            self.set_y(-15)
            # Times italic 8
            self.set_font('Times', 'I', 8)
            # Page number
            self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    pdf = PDF()
    pdf.set_margins(left=20,right=20,top=10)
    pdf.alias_nb_pages()
    pdf.set_font('Times', 'B', 16)

    # Effective page width, or just epw
    epw = pdf.w - 2 * pdf.l_margin

    # Set column width to 1/3 of effective page width to distribute content
    # evenly across table and page
    col_width = epw / 2

    # Text height is the same as current font size
    th = pdf.font_size
    pdf.add_page()

    pdf.set_font('Times', 'B', 30)
    pdf.ln(40)
    pdf.set_x(pdf.l_margin + 15)
    pdf.write(5,"Agora: Automated Genetically")
    pdf.ln(15)
    pdf.set_x(pdf.l_margin + 15)
    pdf.write(5," Optimized Raman Analytics")

    pdf.set_xy(pdf.l_margin, 85)
    pdf.set_font('Times', 'B', 18)
    pdf.write(5, "Requestor Details")

    pdf.set_font('Times', 'B', 12)

    y_set = 100

    pdf.set_xy(pdf.l_margin, y_set)

    for i in range(0, len(req_df)):

        attr = 0

        if attr < 1:

            for j in range(0, len(req_df.columns)):
                pdf.cell(col_width, 1.5 * th, str(req_df.iloc[i, j]), border=1)

                attr += 1

        y_set += 1.5 * th

        pdf.set_xy(pdf.l_margin, y_set)

    return pdf


def write_run_details(pdf, agora_obj):
    """ Pulls out details from the Agora object """

    # Effective page width, or just epw
    epw = pdf.w - 2 * pdf.l_margin

    # Set column width to 1/3 of effective page width to distribute content
    # evenly across table and page
    col_width = epw / 2

    # Text height is the same as current font size
    th = pdf.font_size

    # Line break
    # pdf.ln(130)

    pdf.add_page()

    ##List of attributes analyzed##
    attr_list = agora_obj.attrs

    mt_dict = {
        'Outliers removed from training data?': agora_obj.remove_out_Tr,
        'Outliers removed from validation data?': agora_obj.remove_out_Val,
        'Replicate correction performed?': agora_obj.rep_corr, 'Replicate collapse performed?': agora_obj.average,
        'Validation file used:': agora_obj.req_df.iloc[8, 1],
        'GA # iterations': agora_obj.niter,
        'GA population size': agora_obj.popSize,
        'GA cut off': agora_obj.cutoff,
        'Datetime of Analysis Started:': agora_obj.start,
        'Datetime of Analysis Ended:': agora_obj.end,
        'Analysis time in minutes:': round((agora_obj.end - agora_obj.start).seconds / 60, 3)}

    pdf.set_xy(pdf.l_margin, 50)
    pdf.set_font('Times', 'B', 18)
    pdf.write(5, "Run Details")

    pdf.set_font('Times', 'B', 12)

    y_set = 65

    pdf.set_xy(pdf.l_margin, y_set)

    ###Add first row separately###
    pdf.cell(col_width, 1.5 * th, str("Attributes Analyzed:"), border=1)

    pdf.set_xy(pdf.l_margin + col_width, y_set)

    ##Render loop through attributes and add into table##
    for item in attr_list:
        pdf.cell(col_width, 1.5 * th, str(item), border=1)

        y_set += 1.5 * th

        pdf.set_xy(pdf.l_margin + col_width, y_set)

    ##Reset row coordinates on page
    pdf.set_xy(pdf.l_margin, y_set)

    for item in mt_dict:
        pdf.cell(col_width, 1.5 * th, str(item), border=1)
        pdf.cell(col_width, 1.5 * th, str(mt_dict[item]), border=1)

        y_set += 1.5 * th

        pdf.set_xy(pdf.l_margin, y_set)

    return pdf

def write_nomenclature(pdf):
    x = pdf.l_margin
    spec_y = 5 + 4 * 10

    pdf.add_page()

    pdf.set_xy(x, spec_y)
    pdf.set_font('Times', 'B', 18)
    pdf.write(5, 'Abbreviations and Nomenclature')
    spec_y += 10
    pdf.set_font('Times', '', 14)
    pdf.set_xy(x, spec_y)

    pdf.write(5, 'Agora:\n          Automated Genetically Optimized Raman Analytics. An in-house iterative\n'
                 '          algorithm which uses metaheuristics to converge on a data processing and\n'
                 '          machine learning pipeline to predict attributes from Raman Data.\n\n'
                 '          Provided that the Raman scientists can design an experiment and format the\n'
                 '          data correctly, Agora generates many (hundreds or more) data processing\n'
                 '          and machine learning pipelines and evaluates their effectiveness. Agora\n'
                 '          saves the best "SIMCA model", which has data processing steps and the PLS\n'
                 '          algorithm to associate spectra to attributes. Additionally, Agora saves the\n'
                 '          best "Universal model", which includes the SIMCA pipeline options plus an\n'
                 '          expanded universe of data processing steps and machine learning algorithms.\n\n'
            'Correlation plot:\n'
            '          Plot of reference values vs their predicted values based on spectroscopic data.\n\n'
            'Data Processing Pipeline:\n          A computer program that sequentially processes or transforms data. The\n'
                 '          Agora data processing pipeline end with prediction using machine learning\n'
                 '          models.\n\n'
            'Machine Learning Model:\n          A computer program that runs an algorithm with learned parameters to predict\n'
            '          numbers or classes from input data. The model is the output of training a\n'
            '          machine learning algorithm on data.\n\n'
            'Modeling Data:\n          A dataset used to create a machine learning model. Modeling data is divided\n'
            '          into a training dataset and a testing dataset. The training is further subdivided\n'
            '          into several "folds" of cross validation datasets for hyperparameter tuning.\n'
            '          The testing dataset is used to assess the prediction properties of the resulting\n'
            '          data processing pipeline.\n\n'
            'Model:\n          Multivariate analysis method used to transform multivariate spectroscopic\n'
            '          data into a univariate value. \n\n'
            'PLS:\n          Projection to latent structures, a multivariate dimensionality reduction method\n'
            '          that summarizes the variation in a large number of input parameters into a\n'
            '          smaller number of input parameters called principal components or scores to\n'
            '          avoid overfitting of a univariate response variable.\n\n')

    spec_y = 55
    pdf.add_page()
    pdf.set_xy(x, spec_y)
    pdf.set_font('Times', '', 14)
    pdf.write(5, 'Principal Component:\n          A latent or derived variable from PLS analysis summarizing some form of \n'
            '          variability in the spectroscopic data that correlates to a univariate response\n'
            '          variable.\n\n'
            'Residual:\n          Difference between the reference method value and the model predicted\n'
            '          value.\n\n'
            'RMSE CV:\n          Root mean square error of cross validation.\n\n'
            'RMSE Test:\n          Root mean squared error on the testing dataset of the modeling data.\n\n'
            'RMSE Valid:\n         Root mean squared error on the validation data.\n\n'
            'SIMCA:\n          Abbreviation of the SIMCA 14/15/16 software used to create PLS models.\n\n'
            'SVR:\n          Suport-vector machine regression method.  Support-vector machine constructs\n'
            '          a hyperplane or set of hyperplanes in a high- or infinite-dimensional space,\n'
            '          which can be used for classification, regression, or other tasks like outliers\n'
            '          detection.\n\n'
            'Validation Data:\n          An external dataset used to assess the prediction properties of the data\n'
            '          processing pipeline. Well constructed external datasets are most representative\n'
            '          of "real world" model performance.\n\n'
            'VIP Plot:\n          Variable Importance to the Projection plot. A rank ordered list of\n'
            '          spectroscopic measurement values based on their importance to establishing\n'
            '          the final model fit.')

    return pdf

def write_descr_stats(pdf, agora_obj):
    x = pdf.l_margin
    textHeight = 20
    spec_y = 45
    asp_ratio = 1.75  ##figure aspect ratio, w/h
    h = 47
    w = asp_ratio * h

    #### Training Set Statistics####
    pdf.add_page()

    pdf.set_xy(pdf.l_margin, spec_y)
    pdf.set_font('Times', 'B', 18)
    pdf.write(5, "Descriptive Statistics -- Training Set")
    spec_y += 10
    pdf.image(agora_obj.raw_tr_spec.replace("./", ""), x, spec_y, w, h)  ##Plot raw training spectra
    x += 1.1 * w  ###Move image placement to the right
    # pdf.image(agora_obj.outl_tr_spec, x, spec_y, w, h)  ##Plot Outlier Training Spectra
    pdf.image(agora_obj.exp_var_dir, x, spec_y, w, h)  ## # of PCs vs variance explained
    x = pdf.l_margin
    spec_y += h + textHeight - 10  ##move image placement down

    # Rescale BiPlot Image Dimensions #
    w_f, h_f = image_rescale(h, agora_obj.biplot_tr_dir)  # Set "h" as reference height -- same height of spectra images

    pdf.image(agora_obj.biplot_tr_dir, x + abs(w_f - w) / 2, spec_y, w_f, h_f)  ##Plot Biplot of training spectra

    x += 1.1 * w  ###Move image placement to the right

    w_f, h_f = image_rescale(h, agora_obj.hist_tr_dir)  # Set "h" as reference height -- same height of spectra images

    pdf.image(agora_obj.hist_tr_dir, x + abs(w_f - w) / 2, spec_y, w_f,
              h_f)  # Plot outliers histogram of training spectra

    ####Table for BiPlot legend key###
    epw = pdf.w - 2 * pdf.l_margin  # Set column width to 1/2 of effective page width to distribute content
    col_width = epw / 2  # evenly across table and page
    th = pdf.font_size  # Text height is the same as current font size
    pdf.set_xy(pdf.l_margin, 170)
    ###Add first row separately###
    pdf.cell(col_width, 1.5 * th, "Legend Label", border=1)
    pdf.cell(col_width, 1.5 * th, "Outlier Scan ID", border=1)
    ##Set Font for the Table
    pdf.set_font('Times', 'B', 10)
    # test_list = ["item 1", "item 2", "item 3", "item 4", "item 5",
    #             "item 6", "item 7", "item 8", "item 9", "item 10"]
    # agora_obj.ranked_outliers
    y_set = 170
    y_set += 1.5 * th
    pdf.set_xy(pdf.l_margin, y_set)
    for i, lst_item in enumerate(agora_obj.ranked_train_outlrs):
        pdf.cell(col_width, 1.5 * th, str(i), border=1)
        pdf.cell(col_width, 1.5 * th, str(lst_item), border=1)
        y_set += 1.5 * th

        pdf.set_xy(pdf.l_margin, y_set)

    return pdf


def write_descr_stats_val(pdf, agora_obj):
    x = pdf.l_margin
    textHeight = 20
    spec_y = 45
    asp_ratio = 1.75  ##figure aspect ratio, w/h
    h = 47
    w = asp_ratio * h

    pdf.add_page()

    pdf.set_xy(pdf.l_margin, spec_y)
    pdf.set_font('Times', 'B', 18)
    pdf.write(5, "Descriptive Statistics -- Validation Set")
    spec_y += 10
    pdf.image(agora_obj.raw_val_spec, x, spec_y, w, h)
    x += 1.1 * w  ###Move image placement to the right

    # pdf.image(agora_obj.outl_val_spec, x, spec_y, w, h)  ##Plot Outlier Validation Spectra
    pdf.image(agora_obj.exp_var_dir, x, spec_y, w, h)
    x = pdf.l_margin
    spec_y += h + textHeight - 10  ##move image placement down

    # Rescale BiPlot Image Dimensions #
    w_f, h_f = image_rescale(h,
                             agora_obj.biplot_val_dir)  # Set "h" as reference height -- same height of spectra images

    pdf.image(agora_obj.biplot_val_dir, x + abs(w_f - w) / 2, spec_y, w_f, h_f)  ##Plot Biplot of validation data

    x += 1.1 * w  ###Move image placement to the right

    # Rescale Histogram Image Dimensions #
    w_f, h_f = image_rescale(h, agora_obj.hist_val_dir)  # Set "h" as reference height -- same height of spectra images

    pdf.image(agora_obj.hist_val_dir, x + abs(w_f - w) / 2, spec_y, w_f, h_f)  ##Plot Histogram of validation data
    '''
    pdf.image(agora_obj.hist_val_dir, x + abs(w_f - w)/2, spec_y, w_f, h_f)  ##Plot outliers histogram
    pdf.ln(170)
    pdf.write(5, "Top Possible Outliers (ranked by distance from centroid in descending order):")
    pdf.ln(10)

    if len(agora_obj.ranked_val_outlrs) > 0:
        pdf.write(5, plotting.outlier_to_string(agora_obj.ranked_val_outlrs))  ###Write top ranked outliers

    else:
        pdf.write(5, "No outliers flagged.")
    '''

    ####Table for BiPlot legend key###
    epw = pdf.w - 2 * pdf.l_margin  # Set column width to 1/2 of effective page width to distribute content
    col_width = epw / 2  # evenly across table and page
    th = pdf.font_size  # Text height is the same as current font size
    pdf.set_xy(pdf.l_margin, 170)
    ###Add first row separately###
    pdf.cell(col_width, 1.5 * th, "Legend Label", border=1)
    pdf.cell(col_width, 1.5 * th, "Outlier Scan ID", border=1)
    ##Set Font for the Table
    pdf.set_font('Times', 'B', 10)
    # test_list = ["item 1", "item 2", "item 3", "item 4", "item 5",
    #             "item 6", "item 7", "item 8", "item 9", "item 10"]
    # agora_obj.ranked_outliers
    y_set = 170
    y_set += 1.5 * th
    pdf.set_xy(pdf.l_margin, y_set)
    for i, lst_item in enumerate(agora_obj.ranked_val_outlrs):
        pdf.cell(col_width, 1.5 * th, str(i), border=1)
        pdf.cell(col_width, 1.5 * th, str(lst_item), border=1)
        y_set += 1.5 * th

        pdf.set_xy(pdf.l_margin, y_set)

    return pdf


def write_emsc_stats(pdf, agora_obj):
    x = pdf.l_margin
    textHeight = 20
    spec_y = 45
    asp_ratio = 1.75  ##figure aspect ratio, w/h
    h = 47
    w = asp_ratio * h

    #### Training Set Statistics####
    pdf.add_page()

    pdf.set_xy(pdf.l_margin, spec_y)
    pdf.set_font('Times', 'B', 18)
    pdf.write(5, "Descriptive Statistics -- Training Set post-EMSC Processing")
    spec_y += 10
    pdf.image(agora_obj.raw_tr_spec.replace("./", ""), x, spec_y, w, h)  ##Plot raw training spectra
    x += 1.1 * w  ###Move image placement to the right

    # pdf.image(agora_obj.outl_tr_spec_emsc, x, spec_y, w, h)  ##Plot Outlier Training Spectra
    pdf.image(agora_obj.exp_var_dir, x, spec_y, w, h)
    x = pdf.l_margin
    spec_y += h + textHeight - 10  ##move image placement down

    # Rescale BiPlot Image Dimensions #
    w_f, h_f = image_rescale(h,
                             agora_obj.biplot_tr_dir_emsc)  # Set "h" as reference height -- same height of spectra images

    pdf.image(agora_obj.biplot_tr_dir_emsc, x + abs(w_f - w) / 2, spec_y, w_f, h_f)  ##Plot Biplot of training spectra

    x += 95  ###Move image placement to the right

    # Rescale Histogram Image Dimensions #
    w_f, h_f = image_rescale(h,
                             agora_obj.hist_tr_dir_emsc)  # Set "h" as reference height -- same height of spectra images

    pdf.image(agora_obj.hist_tr_dir_emsc, x + abs(w_f - w) / 2, spec_y, w_f,
              h_f)  # Plot outliers histogram of training spectra

    ''' 
    pdf.ln(170)
    pdf.write(5, "Top Possible Outliers (ranked by distance from centroid in descending order):")
    pdf.ln(10)

    if len(agora_obj.ranked_train_outlrs) > 0:
        pdf.write(5, plotting.outlier_to_string(agora_obj.ranked_train_outlrs)) # Write top ranked outliers

    else:
        pdf.write(5, "No outliers flagged.")
    '''
    ####Table for BiPlot legend key###
    epw = pdf.w - 2 * pdf.l_margin  # Set column width to 1/2 of effective page width to distribute content
    col_width = epw / 2  # evenly across table and page
    th = pdf.font_size  # Text height is the same as current font size
    pdf.set_xy(pdf.l_margin, 170)
    ###Add first row separately###
    pdf.cell(col_width, 1.5 * th, "Legend Label", border=1)
    pdf.cell(col_width, 1.5 * th, "Outlier Scan ID", border=1)
    ##Set Font for the Table
    pdf.set_font('Times', 'B', 10)

    y_set = 170
    y_set += 1.5 * th
    pdf.set_xy(pdf.l_margin, y_set)
    for i, lst_item in enumerate(agora_obj.ranked_train_outlrs):
        pdf.cell(col_width, 1.5 * th, str(i), border=1)
        pdf.cell(col_width, 1.5 * th, str(lst_item), border=1)
        y_set += 1.5 * th

        pdf.set_xy(pdf.l_margin, y_set)

    return pdf


def write_emsc_stats_val(pdf, agora_obj):
    x = pdf.l_margin
    textHeight = 20
    spec_y = 45
    asp_ratio = 1.75  ##figure aspect ratio, w/h
    h = 47
    w = asp_ratio * h

    #### Training Set Statistics####
    pdf.add_page()

    pdf.set_xy(pdf.l_margin, spec_y)
    pdf.set_font('Times', 'B', 18)
    pdf.write(5, "Descriptive Statistics -- Validation Set post-EMSC Processing")
    spec_y += 10
    pdf.image(agora_obj.raw_val_spec.replace("./", ""), x, spec_y, w, h)  ##Plot raw training spectra

    x += 1.1 * w  ###Move image placement to the right
    # pdf.image(agora_obj.outl_val_spec_emsc, x, spec_y, w, h)  ##Plot Outlier Training Spectra
    pdf.image(agora_obj.exp_var_dir, x, spec_y, w, h)
    x = pdf.l_margin
    spec_y += h + textHeight - 10  ##move image placement down

    # Rescale BiPlot Image Dimensions #
    w_f, h_f = image_rescale(h,
                             agora_obj.biplot_val_dir_emsc)  # Set "h" as reference height -- same height of spectra images

    pdf.image(agora_obj.biplot_val_dir_emsc, x + abs(w_f - w) / 2, spec_y, w_f, h_f)  ##Plot Biplot of training spectra

    x += 95  ###Move image placement to the right

    # Rescale Histogram Image Dimensions #
    w_f, h_f = image_rescale(h,
                             agora_obj.hist_val_dir_emsc)  # Set "h" as reference height -- same height of spectra images

    pdf.image(agora_obj.hist_val_dir_emsc, x + abs(w_f - w) / 2, spec_y, w_f,
              h_f)  # Plot outliers histogram of training spectra
    '''
    pdf.ln(170)
    pdf.write(5, "Top Possible Outliers (ranked by distance from centroid in descending order):")
    pdf.ln(10)
    '''
    ####Table for BiPlot legend key###
    epw = pdf.w - 2 * pdf.l_margin  # Set column width to 1/2 of effective page width to distribute content
    col_width = epw / 2  # evenly across table and page
    th = pdf.font_size  # Text height is the same as current font size

    pdf.set_xy(pdf.l_margin, 170)
    ###Add first row separately###
    pdf.cell(col_width, 1.5 * th, "Legend Label", border=1)
    pdf.cell(col_width, 1.5 * th, "Outlier Scan ID", border=1)
    ##Set Font for the Table
    pdf.set_font('Times', 'B', 10)

    # agora_obj.ranked_outliers
    y_set = 170
    y_set += 1.5 * th
    pdf.set_xy(pdf.l_margin, y_set)
    for i, lst_item in enumerate(agora_obj.ranked_val_outlrs):
        pdf.cell(col_width, 1.5 * th, str(i), border=1)
        pdf.cell(col_width, 1.5 * th, str(lst_item), border=1)
        y_set += 1.5 * th

        pdf.set_xy(pdf.l_margin, y_set)

    return pdf


def write_scans(pdf, agora_obj):
    x = pdf.l_margin
    textHeight = 20
    spec_y = 50
    asp_ratio = 1.75  ##figure aspect ratio, w/h
    h = 47
    w = asp_ratio * h

    #### Principal Component Scans for Training Set ####
    pdf.add_page()

    pdf.set_xy(pdf.l_margin, spec_y)
    pdf.set_font('Times', 'B', 18)
    # pdf.write(5, "Principal Component Scans -- Training Set")
    pdf.write(5, "Principal Component Scans")
    spec_y += 15

    pdf.image(agora_obj.tr_pc1_spec, x, spec_y, w, h)  # insert image of PC1 scan
    x += 1.1 * w ###Move image placement to the right
    pdf.image(agora_obj.tr_pc2_spec, x, spec_y, w, h)  # insert image of PC2 scan
    spec_y += 1.2 * h

    if agora_obj.val_set_given:
        pdf.set_xy(pdf.l_margin, spec_y)
        # pdf.write(5, "Principal Component Scans -- Validation Set")
        x = pdf.l_margin
        pdf.image(agora_obj.val_pc1_spec, x, spec_y, w, h)  # insert image of PC1 scan
        x += 1.1 * w  ###Move image placement to the right
        pdf.image(agora_obj.val_pc2_spec, x, spec_y, w, h)  # insert image of PC2 scan

    return pdf


def write_model_details(pdf, agora_obj, model_name, attr):
    spec_y0 = 5 + 5 * 10
    asp_ratio = 1.75  ##figure aspect ratio, w/h
    h = 47
    w = asp_ratio * h

    pdf.add_page()

    x = pdf.l_margin
    spec_y = spec_y0 - 10

    pdf.set_xy(x, spec_y)
    pdf.set_font('Times', 'B', 16)

    if model_name == 'SIMCA' or model_name == 'simca' or model_name == 'Simca':
        plname = getattr(agora_obj, model_name)
        comment_name = 'SIMCA'
        pdf.write(5, str(attr) + " -- SIMCA Analysis")
    else:
        pdf.write(5, str(attr) + " -- Universal Analysis")
        plname = getattr(agora_obj, 'universal')
        comment_name = 'Agora'

    spec_y += 10
    pdf.set_xy(x, spec_y)
    pdf.set_font('Times', '', 14)
    pdf.write(5, 'Correlation Plot, Predicted vs. Measured')
    x += 1.1 * w
    pdf.set_xy(x, spec_y)
    pdf.write(5, 'Absolute Error in Prediction')
    x = pdf.l_margin
    spec_y += 5
    pdf.image(plname[attr].corr_dir, x, spec_y, w, h)
    x += 1.1 * w
    pdf.image(plname[attr].pred_err_dir, x, spec_y, w, h)
    spec_y += 1.2 * h + 5
    x = pdf.l_margin
    pdf.set_xy(x, spec_y)

    detail_dict = {'Baseline Removal:': 'rem_bl', 'Savtizky-Golay Smoothening:': {'Window': 'window', 'Order': 'order'},
                   'Derivative Filter Order:': 'deriv',
                   "Scaling:": 'sc_method', 'ML method:': 'ml_method', 'ML parameters:': 'ml_params',
                   'RMSE CV:': 'fitness_cv'}

    if agora_obj.val_set_given:
        detail_dict['RMSE P/Valid'] = 'fitness_valid'
    epw = pdf.w - 2 * pdf.l_margin # Effective page width
    col_width = epw / 2 #Table column width
    th = pdf.font_size # Text height is the same as current font size
    y_set = spec_y
    pdf.set_xy(pdf.l_margin, y_set)

    ###Add first row separately###
    pdf.cell(col_width, 1.5 * th, "", border=1)
    pdf.set_xy(pdf.l_margin + col_width, y_set)
    pdf.set_font('Times', 'B', 16)
    pdf.cell(col_width, 1.5 * th, "Model Details:", border=1)
    y_set += 1.5 * th

    pdf.set_xy(pdf.l_margin, y_set)
    pdf.set_font('Times', 'B', 12)

    ##Loop through GA model details dictionary and add to table.
    ga_model = plname[attr]
    # adjustments
    if ga_model.rem_bl:
        detail_dict['Baseline Removal:'] = {'ALS Lambda ': 'lam', 'ALS p ': 'p'}
    else:
        detail_dict['Baseline Removal:'] = 'rem_bl'
    scaling_dict = {0: 'SNV row-wise', 1: 'MAX', 2: 'None', 3: 'None'}  # removed L!!
    # Replicate Correction
    if agora_obj.rep_corr:
        pdf.cell(col_width, 1.5 * th, 'Replicate Correction', border=1)

        pdf.cell(col_width, 1.5 * th, 'EMSC', border=1)
        y_set += 1.5 * th

        pdf.set_xy(pdf.l_margin, y_set)

    for key in detail_dict.keys():
        pdf.cell(col_width, 1.5 * th, str(key), border=1)
        if key == 'Scaling:':
            pdf.cell(col_width, 1.5 * th, scaling_dict[ga_model.sc_method], border=1)
        else:
            if isinstance(detail_dict[key], dict):  # savgol or ALS case
                subkeys = list(detail_dict[key].keys())
                cell_value = "{}: {} ".format(str(subkeys[0]),
                                              str(getattr(ga_model, detail_dict[key][subkeys[0]])))
                for sub_key in subkeys[1:]:
                    detail = "{}: {} ".format(str(sub_key),
                                              str(getattr(ga_model, detail_dict[key][sub_key])))
                    cell_value = "{}, {}".format(cell_value, detail)
                pdf.cell(col_width, 1.5 * th, cell_value, border=1)


            else:
                if key == 'Baseline Removal:' and not ga_model.rem_bl:
                    ga_detail = 'None'
                    pdf.cell(col_width, 1.5 * th, str(ga_detail), border=1)
                else:
                    ga_detail = getattr(ga_model, detail_dict[key])
                    if isinstance(ga_detail, dict):  # multiple ml params case
                        subkeys = list(ga_detail.keys())
                        cell_value = "{}: {} ".format(str(subkeys[0]), ga_detail[subkeys[0]])
                        for sub_key in subkeys[1:]:
                            cell_value = "{}, {} ".format(str(cell_value),
                                                          str("{}: {} ".format(str(sub_key), ga_detail[sub_key]))
                                                          )
                        pdf.cell(col_width, 1.5 * th, cell_value, border=1)
                    else:
                        pdf.cell(col_width, 1.5 * th, str(ga_detail), border=1)
        y_set += 1.5 * th

        pdf.set_xy(pdf.l_margin, y_set)

    y_spec = y_set
    y_spec += 5
    pdf.set_xy(pdf.l_margin, y_spec)
    pdf.set_font('Times', 'B', 12)
    pdf.write(5, 'Model Details: ')
    pdf.set_font('Times', '', 12)
    x += 28
    pdf.set_xy(x, y_spec)
    pdf.write(5, 'The table above represents the optimized '+comment_name+' data processing pipeline as chosen by the '+
              'genetic algorithm. The step-wise order in which data was analyzed runs from the top to the bottom of the table.')

    return pdf


def add_model_selection_details(pdf, agora_obj, model_name, attr):
    spec_y0 = 5 + 5 * 10
    asp_ratio = 1.75  ##figure aspect ratio, w/h
    h = 75
    w = asp_ratio * h
    x0 = pdf.w / 2 - w / 2

    pdf.add_page()
    spec_y = spec_y0 - 10
    x = pdf.l_margin

    pdf.set_xy(x, spec_y)
    pdf.set_font('Times', 'B', 16)

    if model_name == 'SIMCA' or model_name == 'simca' or model_name == 'Simca':
        plname = getattr(agora_obj, model_name)
        pdf.write(5, str(attr) + " -- SIMCA Analysis")
    else:
        pdf.write(5, str(attr) + " -- Universal Analysis")
        plname = getattr(agora_obj, 'universal')

    x = x0
    spec_y += 10
    pdf.set_xy(x, spec_y)
    pdf.set_font('Times', '', 14)
    pdf.write(5, 'Model Selection')
    spec_y += 10
    pdf.image(plname[attr].ml_selection_dir, x0, spec_y, w, h)
    spec_y += 1.1 * h - 5
    pdf.set_xy(pdf.l_margin, spec_y)
    pdf.set_font('Times', '', 12)
    pdf.write(5, str('In order to find the optimal parameters of the machine learning model, we perform a K-Fold cross validation (K = %d ). '+
                     'We calculate the training and test scores, their means and standard deviation. The final model '+
                     'is selected using the "one-standard-error" rule - we select the model that is within one standard '+
                     'error of the best score.') % agora_obj.cv)
    spec_y += 25
    pdf.set_xy(x, spec_y)
    pdf.set_font('Times', '', 14)
    pdf.write(5, 'Genetic Algorithm Convergence')
    spec_y += 10
    pdf.image(plname[attr].ga_scores_dir, x0, spec_y, w, h)
    spec_y += 1.1 * h - 5
    pdf.set_xy(pdf.l_margin, spec_y)
    pdf.set_font('Times', '', 12)
    pdf.write(5, str('The Genetic Algorithm Convergence represents the fitness metric value during cross validation (RMSE CV) '+
                     'in the population decreasing with every generation until there can be no improvement indicating an optimal '+
                     'processing pipeline has been found.'))

    return pdf


def add_vip_plots(pdf, agora_obj, model_name, attr):
    spec_y0 = 5 + 5 * 10
    asp_ratio = 1.75  ##aspect ratio, w/h
    h = 75
    w = asp_ratio * h
    x0 = pdf.w / 2 - w / 2

    pdf.add_page()
    spec_y = spec_y0 - 10
    x = pdf.l_margin

    pdf.set_xy(x, spec_y)
    pdf.set_font('Times', 'B', 16)

    if model_name == 'SIMCA' or model_name == 'simca' or model_name == 'Simca':
        plname = getattr(agora_obj, model_name)
        pdf.write(5, str(attr) + " -- SIMCA Analysis")
    else:
        pdf.write(5, str(attr) + " -- Universal Analysis")
        plname = getattr(agora_obj, 'universal')

    spec_y += 10
    x = x0
    pdf.set_xy(x, spec_y)
    pdf.set_font('Times', '', 14)
    pdf.write(5, 'Variable Influence by Intensity')
    spec_y += 10
    pdf.image(plname[attr].vip_spec_dir, x0, spec_y, w, h)
    spec_y += 1.1 * h - 5
    x = pdf.l_margin
    pdf.set_xy(x, spec_y)
    pdf.set_font('Times', '', 12)
    pdf.write(5, str('Variable Importance to the Projection plot. A shading gradient imposed on processed spectra'
                     ' based on the spectroscopic measurements importance to establishing the final model fit'))
    spec_y += 20

    x = x0
    pdf.set_xy(x, spec_y)
    pdf.set_font('Times', '', 14)
    pdf.write(5, 'Variable Influence by Attribute Value')
    spec_y += 10
    pdf.image(plname[attr].vip_bar_dir, x0, spec_y, w, h)
    spec_y += 1.1 * h - 5
    x = pdf.l_margin
    pdf.set_xy(x, spec_y)
    pdf.set_font('Times', '', 12)
    pdf.write(5, str('Variable Importance to the Projection plot. A rank ordered list of spectroscopic measurement'
                     ' values based on their importance to establishing the final model fit'))

    return pdf


def add_raw_vs_processed(pdf, agora_obj, model_name, attr):
    spec_y0 = 5 + 5 * 10
    asp_ratio = 1.75  ##aspect ratio, w/h
    h = 75
    w = asp_ratio * h
    x0 = pdf.w / 2 - w / 2

    pdf.add_page()
    spec_y = spec_y0 - 10
    x = x0

    pdf.set_xy(x, spec_y)
    pdf.set_font('Times', 'B', 16)

    if model_name == 'SIMCA' or model_name == 'simca' or model_name == 'Simca':
        plname = getattr(agora_obj, model_name)
        comment_name = 'SIMCA'
        pdf.set_xy(pdf.l_margin, spec_y)
        pdf.write(5, str(attr) + " -- SIMCA Analysis")
    else:
        plname = getattr(agora_obj, 'universal')
        comment_name = 'Agora'
        pdf.set_xy(pdf.l_margin, spec_y)
        pdf.write(5, str(attr) + " -- Universal Analysis")
    spec_y += 10
    pdf.set_xy(x0, spec_y)
    pdf.set_font('Times', '', 14)
    pdf.write(5, 'Raw Spectra')
    spec_y += 10
    pdf.image(plname[attr].raw_attr_spec_dir, x0, spec_y, w, h)
    spec_y += 1.1 * h - 5
    pdf.set_xy(pdf.l_margin, spec_y)
    pdf.set_font('Times', '', 12)
    if agora_obj.val_set_given:
        fitness_metric_raw = round(plname[attr].fitness_valid_raw,4)
    else:
        fitness_metric_raw = round(plname[attr].fitness_dev_raw, 4)
    fitness_metric = round(plname[attr].fitness_cv, 4)

    caption = ('Raw spectra before '+comment_name+' preprocessing (baseline correction, smoothing, etc.).'+
                     # 'Average signal to noise ratio (SNR) value normalized to highest spectra peak: '+str(round(plname[attr].snr_raw, 4))+
                     '  RMSE value: '+str(fitness_metric_raw))
    pdf.write(5, caption)
    spec_y += 15

    pdf.set_xy(x0, spec_y)
    pdf.set_font('Times', '', 14)
    pdf.write(5, 'Processed Spectra')
    spec_y += 10
    pdf.image(plname[attr].pr_spec_dir, x0, spec_y, w, h)
    spec_y += 1.1 * h - 5
    pdf.set_xy(pdf.l_margin, spec_y)
    pdf.set_font('Times', '', 12)
    caption = (comment_name+' processed spectra.'
                            # 'Average signal to noise ratio (SNR) value normalized to highest spectra peak: ' + str(round(plname[attr].snr, 4)) +
                     '  RMSE value: ' + str(fitness_metric))
    pdf.write(5, caption)

    spec_y += 15
    x = pdf.l_margin
    pdf.set_xy(x, spec_y)
    pdf.set_font('Times', '', 12)
    if fitness_metric < fitness_metric_raw:
        caption = 'Conclusion: preprocessing steps yielded a lower RMSE value.'
    else:
        caption = 'Warning: the processed spectra has a higher RMSE value than raw spectra.'
    # if plname[attr].snr > plname[attr].snr_raw and fitness_metric < fitness_metric_raw:
    #     caption = str('Conclusion: preprocessing steps yielded a lower RMSE value and a higher normalized SNR average.')
    # elif plname[attr].snr > plname[attr].snr_raw and fitness_metric > fitness_metric_raw:
    #     caption = str('Warning: although preprocessing steps yielded a higher normalized SNR average, the processed spectra '+
    #                   'has higher RMSE values than raw spectra.')
    # elif plname[attr].snr < plname[attr].snr_raw and fitness_metric < fitness_metric_raw:
    #     caption = str('Warning: although preprocessing steps yielded a lower RMSE value, the processed spectra '+
    #                   'has a lower normalized SNR average than raw spectra.')
    # else:
    #     caption = str('WARNING: raw spectra has lower RMSE values and higher normalized SNR averages than those of processed spectra.')
    pdf.write(5, caption)

    return pdf


def write_appendix(pdf, agora_obj):
    pdf.add_page()

    Tr_out = agora_obj.all_train_outlrs
    Val_out = agora_obj.all_val_outlrs

    pdf.set_xy(pdf.l_margin, 50)
    pdf.set_font('Times', 'B', 18)
    pdf.write(5, "Appendix:")

    x = pdf.l_margin
    spec_y = 60
    asp_ratio = 1.75  ##figure aspect ratio, w/h
    h = 47
    w = asp_ratio * h

    ###List All Possible Outliers###
    pdf.set_xy(x, spec_y)
    pdf.set_font('Times', 'B', 14)
    pdf.write(5, "All Possible Training Set Outliers:")
    pdf.ln(10)
    pdf.set_font('Times', '', 12)
    if len(Tr_out) > 0:
        pdf.write(5, plotting.outlier_to_string(Tr_out))
    else:
        pdf.set_x(pdf.l_margin)
        pdf.write(5, 'None')
    pdf.ln(10)
    pdf.set_font('Times', 'B', 14)
    pdf.set_x(pdf.l_margin)
    pdf.write(5, "All Possible Validation Set Outliers:")
    pdf.ln(10)
    pdf.set_font('Times', '', 12)
    if len(Val_out) > 0:
        pdf.write(5, plotting.outlier_to_string(Val_out))
    else:
        pdf.write(5, 'None')

    ###Add Influence Plots###
    pdf.add_page()

    pdf.set_xy(pdf.l_margin, 50)
    pdf.set_font('Times', 'B', 18)
    pdf.write(5, "Influence Plots:")
    spec_y += 5

    for attr in agora_obj.attrs.values:
        pdf.set_font('Times', '', 16)
        pdf.set_xy(x, spec_y)
        pdf.write(5, attr)
        spec_y += 10
        pdf.set_xy(x, spec_y)
        pdf.write(5, 'Universal Analysis')
        x += 95
        pdf.set_x(x)
        pdf.write(5, 'SIMCA Analysis')
        x -= 95
        pdf.set_x(x)
        spec_y += 10

        pdf.image(agora_obj.universal[attr].infl_dir, x, spec_y, w, h)
        x += 1.1 * w
        pdf.image(agora_obj.simca[attr].infl_dir, x, spec_y, w, h)

        spec_y += 1.1 * h
        x = pdf.l_margin
        pdf.set_xy(x, spec_y)

    return pdf


def make_appendix_section(pdf, img_list, section_title):
    pdf.add_page()

    x = pdf.l_margin
    textHeight = 20
    spec_y = 5 + 6 * 10
    w = 87.5
    h = 50

    pdf.set_xy(pdf.l_margin, 40)
    pdf.set_font('Times', 'B', 18)
    title = "Appendix" + section_title
    pdf.write(5, title)

    for image in img_list:
        pdf.image(image, x, spec_y, w, h)
        spec_y += h + textHeight  ##move image placement down

    return pdf


def make_pdf(agora_obj):
    ##Add requestor details text
    pdf_doc = write_requestor_details(agora_obj.req_df)
    ##Add run details
    pdf_doc = write_run_details(pdf_doc, agora_obj)
    ###Add page(s) of nomenclature and abbreviations
    pdf_doc = write_nomenclature(pdf_doc)
    ###Add descriptive statistics section###
    pdf_doc = write_descr_stats(pdf_doc, agora_obj)
    if agora_obj.rep_corr:
        write_emsc_stats(pdf_doc, agora_obj)
    ##Add descriptive statistics section for validation data
    if agora_obj.val_set_given:
        pdf_doc = write_descr_stats_val(pdf_doc, agora_obj)
        if agora_obj.rep_corr:
            write_emsc_stats_val(pdf_doc, agora_obj)
    ###Add attribute-specific model detail sections###
    for attr in agora_obj.attrs.values:
        pdf_doc = write_model_details(pdf_doc, agora_obj, 'simca', attr)
        ####Place holder for when simca figures are added####
        #     pdf_doc = add_model_selection_details(pdf_doc, agora_obj, 'simca', attr)
        #     pdf_doc = add_vip_plots(pdf_doc, agora_obj, 'simca', attr)
        pdf_doc = add_raw_vs_processed(pdf_doc, agora_obj, 'simca', attr)
        pdf_doc = write_model_details(pdf_doc, agora_obj, 'agora', attr)
        pdf_doc = add_raw_vs_processed(pdf_doc, agora_obj, 'agora', attr)
        pdf_doc = add_model_selection_details(pdf_doc, agora_obj, 'agora', attr)
        if os.path.exists(agora_obj.universal[attr].vip_bar_dir) and os.path.exists(agora_obj.universal[attr].vip_spec_dir):
            pdf_doc = add_vip_plots(pdf_doc, agora_obj, 'agora', attr)

    pdf_doc = write_appendix(pdf_doc, agora_obj)
    pdf_doc = write_scans(pdf_doc, agora_obj)
    pdf_doc.output(f"{agora_obj.plot_dir}/{str(date.today())}_AutoReport.pdf", "F")

    return