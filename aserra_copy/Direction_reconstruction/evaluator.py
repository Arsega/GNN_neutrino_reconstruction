#!/usr/bin/env python
# coding: utf-8

import numpy as np
#from radiotools import plthelpers as php
#from matplotlib import pyplot as plt

import os
import time
from radiotools import helper as hp
from NuRadioReco.utilities import units
import pickle
import argparse
from sklearn.manifold import TSNE

#from termcolor import colored
from toolbox import load_file, models_dir
from constants import datapath, data_filename, label_filename, test_file_ids
# -------

# GPU allocation
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using",device,". GPU # is",torch.cuda.current_device())
# --------------


run_id='direction_base_all_higher'
# Save the run name and filename
run_name = f"run{run_id}"
filename = f"model_history_log_{run_name}.csv"

# Models folder
saved_model_dir = models_dir(run_name)

print(f"Evaluating energy resolution for {run_name}")

# Load the model
from generator import GCN
mymodel = GCN().to(device)
save_model_path=os.path.join(saved_model_dir, f"{run_name}.pt")
#save_model_path=os.path.join(saved_model_dir, f"latest_model_checkpoint-v3.ckpt")
mymodel.load_state_dict(torch.load(save_model_path))
#checkpoint = torch.load(save_model_path)
#mymodel.load_state_dict(checkpoint['state_dict'])
mymodel.float
mymodel.eval()


# Load test file data and make predictions
from generator import Prepare_Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader

from constants import test_data_points, n_files, n_files_val

n_files_test = 4
n_files_train = n_files - n_files_val - n_files_test
list_of_file_ids_test = np.arange(n_files_test, dtype=int)
list_of_file_ids_test_small = np.random.choice(list_of_file_ids_test, size=4, replace=False)  # train_files

test = Prepare_Dataset(list_of_file_ids_test_small, points=99000) # files id [40], points=30000
x_test, y_test = test[0]
print("Length of training dataset: ", len(test), x_test.shape, y_test)

test_loader = DataLoader(test, batch_size=1, shuffle=False)


## Evalutate test dataset
import torch.nn.functional as F
list_x = []
list_y = []
list_z = []
dire_x = []
dire_y = []
dire_z = []
with torch.no_grad():
    # Iterate through test set minibatchs 
    for x, y in tqdm(test_loader):
        x = x.to(device)
        y = y[0]
        dire_x.append(y[:,0].item())
        dire_y.append(y[:,1].item())
        dire_z.append(y[:,2].item())
        # Forward pass
        pred_e =  mymodel(x) 
        pred_e = torch.squeeze(pred_e[0])
        list_x.append(pred_e[0].item())
        list_y.append(pred_e[1].item())
        list_z.append(pred_e[2].item())


# Save predicted values
dire_x = np.array(dire_x)
dire_y = np.array(dire_y)
dire_z = np.array(dire_z)
dire_x_pred = np.array(list_x)
dire_y_pred = np.array(list_y)
dire_z_pred = np.array(list_z)

theta, phi = hp.cartesian_to_spherical(dire_x, dire_y, dire_z)
theta_pred, phi_pred = hp.cartesian_to_spherical(dire_x_pred, dire_y_pred, dire_z_pred)

with open(f'{saved_model_dir}/model.{run_name}.h5_predicted.pkl', "bw") as fout:
    pickle.dump([theta, phi, theta_pred, phi_pred], fout, protocol=4)


print(f"Done evaluating energy resolution for {run_name}!")
print("")

