import datasets

project_name = "GNN_MSE_direction"
run_version = "test1"
dataset_name = "SouthPole"

# Dataset setup
# Call Dataset(dataset_name, em, noise) with
#     dataset_name:
#         ALVAREZ (only had + noise) / ARZ
#     em: (True means em+had, False means had)
#         True / False (default)
#     noise:
#         True (default) / False
dataset_name = "ARZ"
dataset_em = False
dataset_noise = True

dataset = datasets.Dataset(dataset_name, dataset_em, dataset_noise)

# Paths
datapath = dataset.datapath
data_filename = dataset.data_filename
label_filename = dataset.label_filename

# numbers
n_files = dataset.n_files
n_files_val = dataset.n_files_val
test_file_ids = dataset.test_file_ids

#running stuff
train_files = 3
val_files = 1
test_files = 1
norm = 1e-6
train_data_points = 199900  #190000
val_data_points = 10000 #25000
test_data_points = 10000 #20000
epochs = 100

# Directories
plots_dir = "plots_GNN_MSE_direction"
