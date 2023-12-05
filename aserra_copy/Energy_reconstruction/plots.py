# Imports
import numpy as np
from radiotools import plthelpers as php
from matplotlib import pyplot as plt
import os
import time
import pickle
from scipy import stats
from radiotools import helper as hp
from NuRadioReco.utilities import units
from toolbox import load_file, calculate_percentage_interval, get_pred_energy_diff_data, models_dir, get_histogram2d, median, calculate_percentage_interval_2
import argparse
from constants import datapath, data_filename, label_filename, plots_dir
from scipy.optimize import curve_fit
# -------


run_id='CNN4_GNN4_all_files_4G_2'
# Save the run name and filename
run_name = f"run{run_id}"
filename = f"model_history_log_{run_name}.csv"

print(f"Plotting energy resolution for {run_name}...")

plots_dir=f"aserra/{plots_dir}/{run_id}"
# Make sure plots folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Models folder
saved_model_dir = models_dir(run_name)

loss_file = f'{saved_model_dir}/lightning_logs/version_1/metrics.csv'
import pandas as pd
loss_data = pd.read_csv(loss_file, keep_default_na=False, na_values=' ')

#loss_data = loss_data.groupby(['epoch']).mean()

train = pd.DataFrame(loss_data, columns= ['train_loss'])
val = pd.DataFrame(loss_data, columns= ['val_loss'])
epoch = pd.DataFrame(loss_data, columns= ['epoch'])

train_loss_list = []
val_loss_list = []
train_epoch_list =[]
val_epoch_list = []

train_loss_list = []
val_loss_list = []

for i in range(len(train)+1):
    if np.mod(i,2) == 1 :
        # print(i, np.mod(i,2),train['train_loss'][i])
        train_loss_list.append(float(train['train_loss'][i]))

for i in range(len(val)):
    if np.mod(i,2) == 0 :
        # print(i, np.mod(i,2),val['val_loss'][i])
        val_loss_list.append(float(val['val_loss'][i]))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot( train_loss_list[1::])
ax.plot( val_loss_list[1::])
ax.set_xlabel("epochs -1")
ax.set_ylabel("loss")
ax.legend(['train', 'val'], loc='upper right')
#fig.savefig(f"{plots_dir}/loss_{run_name}_new.png")


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(train_loss_list[1::])
ax.plot(val_loss_list[1::])
ax.set_yscale('log')
ax.set_xlabel("epochs -1")
ax.set_ylabel("loss")
ax.legend(['train', 'val'], loc='upper right')
#fig.savefig(f"{plots_dir}/loss_log_{run_name}_new.png")


# Get angle difference data
energy_difference_data = get_pred_energy_diff_data(run_name)
print(energy_difference_data)
ave = np.average(energy_difference_data)
variance = np.average((energy_difference_data - ave)**2)
std = variance**0.5
print('standard deviation= ', std)

# Redefine N
N = energy_difference_data.size

# Calculate 68 %
q1 = calculate_percentage_interval_2(energy_difference_data, 0.16)
q2 = calculate_percentage_interval_2(energy_difference_data, 0.84)
media = median(energy_difference_data)
print('68+= ',np.abs(media - q2))
print('68-= ', np.abs(media - q1))
energy_68 = calculate_percentage_interval(energy_difference_data, 0.68)
print(energy_68)


delta_log_E_string = r"$\Delta(\log_{10}\:E)$"
fig, ax = php.get_histogram(energy_difference_data, bins=np.linspace(-1.5, 1.5, 90),
                            xlabel=delta_log_E_string)
plt.title("Energy resolution for the full dataset")
fig.savefig(f"{plots_dir}/energy_resolution_{run_name}.png")


print(f"Saved energy resolution for {run_name}!")
print("")



#Heat map for energy
#####################
plot_title = f"Heatmap of predicted and true energy for the full data set"
xlabel = f"True  energy"
ylabel = f"Predicted energy"
cmap = "BuPu"
bins = 50

a, shower_energy_log10_predict, shower_energy_log10 = get_pred_energy_diff_data(run_name, do_return_data=True)


index = np.argwhere(shower_energy_log10_predict > 19)
shower_energy_log10_predict = np.delete(shower_energy_log10_predict, index)
shower_energy_log10 = np.delete(shower_energy_log10, index)
energy_difference_data = np.delete(energy_difference_data, index)

index = np.argwhere(shower_energy_log10 < 16.5)
shower_energy_log10_predict = np.delete(shower_energy_log10_predict, index)
shower_energy_log10 = np.delete(shower_energy_log10, index)
energy_difference_data = np.delete(energy_difference_data, index)



for cscale in ["linear", "log"]:
    file_name = f"plots/scatter_2dhistogram_{run_name}_cscale{cscale}.png"
    
    # Also plot a heatmap of the scatter plot instead of just dots
    fig, ax, im = get_histogram2d(shower_energy_log10, shower_energy_log10_predict, fname=file_name, 
                                  title=plot_title, xlabel=xlabel, ylabel=ylabel, bins=bins, 
                                  cmap=cmap, cscale=cscale, normed='colum')

    # Plot a black line through the middle
    xmin = min(shower_energy_log10)
    xmax = max(shower_energy_log10)
    ymin = min(shower_energy_log10_predict)
    ymax = max(shower_energy_log10_predict)

    ax.plot([min(xmin, ymin), max(xmax, ymax)], [min(xmin, ymin), max(xmax, ymax)], 'k--')

    plt.tight_layout()
    plt.xlim([16.5,19])
    plt.ylim([16.5,19])
    plt.savefig(f"{plots_dir}/predicted_energy_vs_true_energy_{cscale}_{run_name}.png", dpi=300)



# energy bins

prediction_file = f'/home/aserra/aserra/GNN_MSE/models/run{run_id}/model.run{run_id}.h5_predicted.pkl'
with open(prediction_file, "br") as fin:
    shower_energy_log10_predict, shower_energy_log10 = pickle.load(fin)

shower_energy_log10_predict = np.squeeze(shower_energy_log10_predict)


energy_difference_data = np.array([ shower_energy_log10_predict[i] - shower_energy_log10[i] for i in range(len(shower_energy_log10))])


arr = np.column_stack((shower_energy_log10, energy_difference_data))
arr = arr[arr[:,0].argsort()]
energy_bin = np.round(np.arange(16.5,19,0.1),2)
split_at = arr[:,0].searchsorted(energy_bin)
arr_split = np.split(arr, split_at)

sigma = []
sig_68 = []
mu = []


for i in range(1,len(energy_bin)):
    mean = arr_split[i][:,1].mean()
    std = arr_split[i][:,1].std()

    # Redefine N
    N = arr_split[i][:,1].size

    # Calculate 68 %
    sigma.append(std)
    mu.append(mean)
    q1 = calculate_percentage_interval_2(arr_split[i][:,1], 0.16)
    q2 = calculate_percentage_interval_2(arr_split[i][:,1], 0.84)
    media = median(arr_split[i][:,1])
    pos_68 = np.abs(media - q2)
    neg_68 = np.abs(media - q1)
    sig_68.append((pos_68+neg_68)*0.5)


fig = plt.figure()
plt.scatter(energy_bin[1:]+0.05, sigma, label='STD')
plt.scatter(energy_bin[1:]+0.05, sig_68, label=r'$\sigma_{68} %$')
plt.ylabel(f'STD bin ' r'$\Delta(\log_{10}\:E)$')
plt.xlabel(f'True shower energy ' r'$(\log_{10}\:E)$')
plt.grid()
plt.legend()
plt.title('Full data set')
fig.savefig(f"{plots_dir}/energy_STD_bin_{run_name}.png")

fig = plt.figure()
plt.scatter(energy_bin[1:]+0.05, mu)
plt.ylabel(f'Mean bin ' r'$\Delta(\log_{10}\:E)$')
plt.xlabel(f'True shower energy ' r'$(\log_{10}\:E)$')
plt.grid()
plt.title('Full data set')
fig.savefig(f"{plots_dir}/energy_mean_bin_{run_name}.png")
    
fig = plt.figure()
plt.scatter(energy_bin[1:]+0.05, sig_68)
plt.ylabel(f'\n68 % interval bin' r'$\Delta(\log_{10}\:E)$')
plt.xlabel(f'True shower energy' r'$(\log_{10}\:E)$')
plt.grid()
plt.title('Full dataset ')
fig.savefig(f"{plots_dir}/energy_68_bin_{run_name}.png")


