import unittest
import os
from agora_lib.agora import web_app_run
project_dir = os.path.dirname(os.path.abspath(__file__))

def local_run_on_model(train_file_dir, val_file_dir='',run_tag='unit_test_',rep_corr=False, average=False, model='ga'):
    if model == 'ga':
        web_app_run(file_dir=train_file_dir, file_dir_test=val_file_dir, plot_dir=project_dir, run_tag=run_tag,rep_corr=rep_corr, average=average,niter=1, popSize=2,
              cutoff=2)
    else:
        web_app_run(file_dir=train_file_dir, file_dir_test=val_file_dir, plot_dir=project_dir, run_tag=run_tag,rep_corr=rep_corr, average=average)


def run_only_train():
    run_tag='Scaled_ph_data_test_'
    train_file_dir = f"{project_dir}/data/scaled_ph_data/Agora_Submission_Test.xlsx"
    local_run_on_model(train_file_dir, run_tag=run_tag)


def run_only_train_two_attrs():
    run_tag='Only_train_two_pac_attrs_test_'
    train_file_dir = f"{project_dir}/data/pac/pH_pac_Training.xlsx"
    local_run_on_model(train_file_dir, run_tag=run_tag)

def run_only_train_full_simca(model='simca'):
    train_file_dir = f"{project_dir}/data/Agora_Submission_Test.xlsx"
    local_run_on_model(train_file_dir, model=model)


def run_train_val_rep_ids_simca(model='simca'):
    train_file_dir = f"{project_dir}/data/Rep_Corr_Training.xlsx"
    val_file_dir = f"{project_dir}/data/Validation.xlsx"
    local_run_on_model(train_file_dir, val_file_dir=val_file_dir, model=model)


def run_only_train_rep_ids_average():
    run_tag='Rep_corr_average_test_'
    train_file_dir = f"{project_dir}/data/replicate_correction/rep_correction_short.xlsx"
    local_run_on_model(train_file_dir, run_tag=run_tag,rep_corr=True, average=True)

def run_train_val_rep_ids_average():
    run_tag='Rep_corr_average_test_'
    train_file_dir = f"{project_dir}/data/replicate_correction/rep_correction_short.xlsx"
    val_file_Dir=f"{project_dir}/data/replicate_correction/rep_correction_short.xlsx"
    local_run_on_model(train_file_dir,val_file_dir=val_file_Dir, run_tag=run_tag,rep_corr=True, average=True)

def run_only_train_rep_ids_not_average():
    run_tag='Rep_corr_not_average_test_'
    train_file_dir = f"{project_dir}/data/replicate_correction/rep_correction_short.xlsx"
    local_run_on_model(train_file_dir,run_tag=run_tag, rep_corr=True, average=False)

def run_train_validate_offlineDoe():
    run_tag='Train_validate_offlineDoe_'
    train_file_dir = f"{project_dir}/data/offlineDoe/AgoraSubmit_Study15_Train_.xlsx"
    val_file_dir=f"{project_dir}/data/offlineDoe/AgoraSubmit_Study15_Val_.xlsx"
    local_run_on_model(train_file_dir=train_file_dir,val_file_dir= val_file_dir, run_tag=run_tag)

def run_glucose_train_only():
    run_tag='Glucose_train_only'
    train_file_dir = f"{project_dir}/data/glucose_data/Glucose_Training.xlsx"
    local_run_on_model(train_file_dir=train_file_dir,run_tag=run_tag)

def run_glucose_train_valid():
    run_tag='Glucose_train_only'
    train_file_dir = f"{project_dir}/data/glucose_data/Glucose_Training.xlsx"
    val_file_dir=f"{project_dir}/data/glucose_data/Glucose_Validation.xlsx"
    local_run_on_model(train_file_dir=train_file_dir, val_file_dir=val_file_dir, run_tag=run_tag)


class TestAgora(unittest.TestCase):

    def test_only_train_short(self):
        self.assertEqual(run_only_train(), None, "Error Occurred")

    def test_only_train_full(self):
        self.assertEqual(run_only_train_two_attrs(), None, "Error Occurred")

    def test_only_train_rep_ids_average(self):
        self.assertEqual(run_only_train_rep_ids_average(), None, "Error Occurred")

    def test_only_train_rep_ids_not_average(self):
        self.assertEqual(run_only_train_rep_ids_not_average(), None, "Error Occurred")

    def test_train_validate_offlineDoe(self):
        self.assertEqual(run_train_validate_offlineDoe(), None, "Error Occurred")

    def test_glucose_train_only(self):
        self.assertEqual(run_glucose_train_only(), None, "Error Occurred on Glucose Train")

    def test_glucose_train_valid(self):
        self.assertEqual(run_glucose_train_valid(), None, "Error Occurred on Glucose Train and Validation")



