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
from toolbox import load_file, calculate_percentage_interval, get_pred_direction_diff_data, models_dir, get_histogram2d, get_histogram
import argparse
from constants import datapath, data_filename, label_filename, plots_dir
from scipy.optimize import curve_fit
# -------

run_id='direction_base_all_higher'
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
fig.savefig(f"{plots_dir}/loss_{run_name}_new.png")


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(train_loss_list[1::])
ax.plot(val_loss_list[1::])
ax.set_yscale('log')
ax.set_xlabel("epochs -1")
ax.set_ylabel("loss")
ax.legend(['train', 'val'], loc='upper right')
fig.savefig(f"{plots_dir}/loss_log_{run_name}_new.png")


#----------------------------------------
# Get angle difference data
direction_difference, azimuth_pred, zenith_pred, azimuth, zenith = get_pred_direction_diff_data(run_name)
ave = np.average(direction_difference)
variance = np.average((direction_difference - ave)**2)
std = variance**0.5
print('standard deviation direction= ', std)

# Redefine N
N = direction_difference.size

# Calculate 68 %
dir_68 = calculate_percentage_interval(direction_difference, 0.68)
print(dir_68)

delta_direction_string = r"$\Delta\Psi(^\circ)$"

def moff(x, a, sig, gamm):
    y = (a*x/(2*np.pi*sig**2))*(1-1/gamm)*(1+x**2/(2*gamm*sig**2))**(-gamm)
    return y

hist_y, hist_x = np.histogram(direction_difference, bins=np.linspace(0, 20, 100))
bin_center = hist_x[:-1] + np.diff(hist_x)/2
popt, cov = curve_fit(moff, bin_center, hist_y)

fig, ax = get_histogram(direction_difference, bins=np.linspace(0, 20, 100),
                            xlabel=delta_direction_string, stats=True, fit=popt)

x_fit_interval = np.linspace(bin_center[0], bin_center[-1], 100)
plt.plot(x_fit_interval, moff(x_fit_interval, *popt), label='fit')
plt.title(f"Direction resolution for the full dataset")
fig.savefig(f"{plots_dir}/Direction_resolution_{run_name}.png")
print(popt)

print(f"Saved direction resolution for {run_name}!")
print("")

#---------------------------------------------
# Get angle difference azimuth data
azimuth_diff = np.array([ azimuth_pred[i] - azimuth[i] for i in range(len(azimuth))]) /units.deg


ave = np.average(azimuth_diff)
variance = np.average((azimuth_diff - ave)**2)
std = variance**0.5
print('standard deviation azimuth= ', std)

# Redefine N
N = azimuth_diff.size

# Calculate 68 %
dir_68 = calculate_percentage_interval(azimuth_diff, 0.68)
print(dir_68)

delta_azimuth_string = r"$\Delta\Phi(^\circ)$"
fig, ax = php.get_histogram(azimuth_diff, bins=np.linspace(-20, 20, 100),
                            xlabel=delta_azimuth_string, stats=False)
plt.title(f"Azimuth resolution for the full dataset")
fig.savefig(f"{plots_dir}/Azimuth_resolution_{run_name}.png")


print(f"Saved direction resolution for {run_name}!")
print("")



#Heat map for energy
#####################
plot_title = f"Heatmap of predicted and true azimuth for the full dataset"
xlabel = f"True  azimuth"
ylabel = f"Predicted azimuth"
cmap = "BuPu"
bins = 100



azimuth = azimuth /units.deg 
azimuth_pred = azimuth_pred /units.deg 

for cscale in ["linear", "log"]:
    file_name = f"plots/scatter_2dhistogram_{run_name}_cscale{cscale}.png"
    
    # Also plot a heatmap of the scatter plot instead of just dots
    fig, ax, im = get_histogram2d(azimuth, azimuth_pred, fname=file_name, 
                                  title=plot_title, xlabel=xlabel, ylabel=ylabel, bins=bins, 
                                  cmap=cmap, cscale=cscale, normed='colum')

    # Plot a black line through the middle
    xmin = min(azimuth)
    xmax = max(azimuth)
    ymin = min(azimuth_pred)
    ymax = max(azimuth_pred)

    ax.plot([min(xmin, ymin), max(xmax, ymax)], [min(xmin, ymin), max(xmax, ymax)], 'k--')

    plt.tight_layout()
    plt.xlim([-180,180])
    plt.ylim([-180,180])
    plt.savefig(f"{plots_dir}/predicted_azimuth_vs_true_azimuth_{cscale}_{run_name}.png", dpi=300)




#--------------------------------------

# Get angle difference zenith data
zenith_diff = np.array([ zenith_pred[i] - zenith[i] for i in range(len(zenith))]) /units.deg


ave = np.average(zenith_diff)
variance = np.average((zenith_diff - ave)**2)
std = variance**0.5
print('standard deviation zenith= ', std)

# Redefine N
N = zenith_diff.size

# Calculate 68 %
dir_68 = calculate_percentage_interval(zenith_diff, 0.68)
print(dir_68)

delta_zenith_string = r"$\Delta\Theta(^\circ)$"
# fig, ax = php.get_histogram(predicted_nu_energy[:, 0], bins=np.arange(17, 20.1, 0.05), xlabel="predicted energy")
fig, ax = php.get_histogram(zenith_diff, bins=np.linspace(-20, 20, 100),
                            xlabel=delta_zenith_string)
# ax.plot(xl, N*stats.rayleigh(scale=scale, loc=loc).pdf(xl))
plt.title(f"Zenith resolution for the full dataset")
fig.savefig(f"{plots_dir}/Zenith_resolution_{run_name}.png")


print(f"Saved zenith resolution for {run_name}!")
print("")



#Heat map for energy
#####################
plot_title = f"Heatmap of predicted and true zenith for the full dataset"
xlabel = f"True  zenith"
ylabel = f"Predicted zenith"
cmap = "BuPu"
bins = 100



zenith = zenith /units.deg 
zenith_pred = zenith_pred /units.deg 

for cscale in ["linear", "log"]:
    file_name = f"plots/scatter_2dhistogram_{run_name}_cscale{cscale}.png"
    
    # Also plot a heatmap of the scatter plot instead of just dots
    fig, ax, im = get_histogram2d(zenith, zenith_pred, fname=file_name, 
                                  title=plot_title, xlabel=xlabel, ylabel=ylabel, bins=bins, 
                                  cmap=cmap, cscale=cscale, normed='colum1')

    # Plot a black line through the middle
    xmin = min(zenith)
    xmax = max(zenith)
    ymin = min(zenith_pred)
    ymax = max(zenith_pred)

    ax.plot([min(xmin, ymin), max(xmax, ymax)], [min(xmin, ymin), max(xmax, ymax)], 'k--')

    plt.tight_layout()
    plt.xlim([40,140])
    plt.ylim([40,140])
    plt.savefig(f"{plots_dir}/predicted_zenith_vs_true_azimuth_{cscale}_{run_name}.png", dpi=300)



# direction bins

direction = hp.spherical_to_cartesian(zenith, azimuth)

arr = np.column_stack((azimuth, direction_difference))
arr = arr[arr[:,0].argsort()]
direction_bin = np.round(np.arange(-160,160,20),2)
split_at = arr[:,0].searchsorted(direction_bin)
arr_split = np.split(arr, split_at)
sigma = []
mu = []
sig_68 = []

for i in range(1,len((direction_bin))):
 
    # Calculate 68 %
    energy_68 = calculate_percentage_interval(arr_split[i][:,1], 0.68)
    sig_68.append(energy_68)


fig = plt.figure()
plt.scatter(direction_bin[1:]+10, sig_68)
plt.ylabel(f'\n68 % interval bin')
plt.xlabel(f'True azimuth')
plt.grid()
plt.ylim(0,30)
plt.title('Full dataset azimuth')
fig.savefig(f"{plots_dir}/azimuth_68_bin_{run_name}.png")



arr = np.column_stack((zenith, direction_difference))
arr = arr[arr[:,0].argsort()]
direction_bin = np.round(np.arange(50,150,10),2)
split_at = arr[:,0].searchsorted(direction_bin)
arr_split = np.split(arr, split_at)
sigma = []
mu = []
sig_68 = []

for i in range(1,len((direction_bin))):

    # Calculate 68 %
    energy_68 = calculate_percentage_interval(arr_split[i][:,1], 0.68)
    sig_68.append(energy_68)


    
fig = plt.figure()
plt.scatter(direction_bin[1:]+5, sig_68)
plt.ylabel(f'\n68 % interval bin' )
plt.xlabel(f'True zenith')
plt.grid()
plt.title('Full dataset zenith')
plt.ylim(0,30)
fig.savefig(f"{plots_dir}/zenith_68_bin_{run_name}.png")
