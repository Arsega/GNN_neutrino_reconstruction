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


run_id='CNN4_GNN4_all_files_4G_2'
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
#save_model_path=os.path.join(saved_model_dir, f"latest_model_checkpoint-v15.ckpt")
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
list_of_file_ids_test = np.arange(n_files_train +n_files_val,n_files, dtype=int)
list_of_file_ids_test_small = np.random.choice(list_of_file_ids_test, size=4, replace=False)  # train_files
print(list_of_file_ids_test)
test = Prepare_Dataset(list_of_file_ids_test, points=99000) # files id [40], points=30000
x_test, y_test = test[0]
print("Length of training dataset: ", len(test), x_test.shape, y_test)
test_loader = DataLoader(test, batch_size=1, shuffle=False)


## Evalutate test dataset
import torch.nn.functional as F
x_list=[]
shower_energy_log10 = []
with torch.no_grad():
    # Iterate through test set minibatchs 
    for x, y in tqdm(test_loader):
        x = x.to(device)
        shower_energy_log10.append(y.item())
        # Forward pass
        pred_e =  mymodel(x) 
        pred_e = torch.squeeze(pred_e[0])
        x_list.append(pred_e.item())


# Save predicted values
shower_energy_log10 = np.array(shower_energy_log10)
shower_energy_log10_predict = np.array(x_list)
with open(f'{saved_model_dir}/model.{run_name}.h5_predicted_test.pkl', "bw") as fout:
    pickle.dump([shower_energy_log10_predict, shower_energy_log10], fout, protocol=4)
#xs, ys = zip(*TSNE().fit_transform(embedding.detach().cpu().numpy()))
#embe = np.column_stack((xs,ys))

from sklearn.decomposition import PCA
import csv




print(f"Done evaluating energy resolution for {run_name}!")
print("")

