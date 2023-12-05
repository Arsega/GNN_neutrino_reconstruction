# Test code for a graph cnn using pytorch lightning

import time
import os
import argparse
import numpy as np
import torch
from torch.nn import Dropout, ReLU, ModuleList, MSELoss, AvgPool2d, Conv2d, BatchNorm2d, Module, GaussianNLLLoss
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv, TopKPooling, BatchNorm, Linear, GraphNorm, EdgePooling, EdgeConv, DynamicEdgeConv, global_mean_pool, avg_pool_neighbor_x, avg_pool_x
from torch_geometric.nn.pool import knn_graph
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum
#from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
#from torch_geometric.nn import GCNConv #, Sequentail, global_mean_pool
import pytorch_lightning as pl 
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from constants import run_version, dataset_name, datapath, data_filename, label_filename, plots_dir, project_name, n_files, n_files_val, dataset_em, dataset_noise, test_file_ids, train_data_points, val_data_points, test_data_points, epochs, train_files, val_files
from toolbox import load_file, find_68_interval, models_dir
from generator import Prepare_Dataset, Prepare_Dataset_val
from torch_geometric.data import Data



device = torch.device("cuda")

#Values for training
architectures_dir = "architectures"
learning_rate = 0.00003 #0.00005 0.000045
es_patience = 7
es_min_delta = 0.0001 # Old value: es_min_delta = 0.0001
norm = 1e-6
batchSize = 64 #64
criterion = MSELoss()

# ------


run_id='CNN4_GNN4'

# Save the run name
run_name = f"run{run_id}"


# Models folder
saved_model_dir = models_dir(run_name)

# Make sure saved_models folder exists
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)

# Make sure architectures folder exists
if not os.path.exists(f"{saved_model_dir}/{architectures_dir}"):
    os.makedirs(f"{saved_model_dir}/{architectures_dir}")

# Make sure plot dir exists
plots_dir=f"aserra/{plots_dir}/{run_id}"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

#constnat for dataset
print("datapath = ",datapath)
print("data_filename = ", data_filename)
print("label_filename = ", label_filename)


n_files_test = 4
n_files_train = n_files - n_files_val - n_files_test

list_of_file_ids_train = np.arange(n_files_train, dtype=int)
list_of_file_ids_val = np.arange(n_files_train, n_files_train + n_files_val, dtype=int)#np.int?
list_of_file_ids_test = np.arange(n_files_train + n_files_val, n_files, dtype=int)
n_events_per_file = 100000
print("Training files #: ",list_of_file_ids_train)
print("Val files #: ",list_of_file_ids_val)
print("Test files #: ",list_of_file_ids_test)

#dataset
print("Preparing datasets...")
# Randomly choose one/some files in the list, size = number of file you want
list_of_file_ids_train_small = np.random.choice(list_of_file_ids_train, size=31, replace=False)  # train_files
print("Picked training set ids:",list_of_file_ids_train_small)
list_of_file_ids_val_small = np.random.choice(list_of_file_ids_val, size=6, replace=False)  # val_files
print("Picked val set ids:",list_of_file_ids_val_small)
#list_of_file_ids_test_small = np.random.choice(list_of_file_ids_test, size=1, replace=False)



train = Prepare_Dataset(list_of_file_ids_train_small, points=99000)
print('train data done')

dataloader_train = DataLoader(train, batch_size=batchSize, shuffle=False, num_workers=0)
print('train dataload done')

val = Prepare_Dataset(list_of_file_ids_val_small, points=99000)

dataloader_val = DataLoader(val, batch_size=1000, shuffle=False, num_workers=0)


# Model params
conv2D_filter_size = 5
pooling_size = 4
amount_Conv2D_layers_per_block = 3 
amount_Conv2D_blocks = 4
conv2D_filter_amount = 32
neighbors = 4

   #self.hidden = ModuleList()
        #Layers = [1280,1024,1024,512,256,128]
        #for input_size, output_size in zip(Layers, Layers[1:]):
        #    self.hidden.append(Linear(input_size, output_size))

def count_params(module):
    sum([np.prod(p.shape) for p in module.parameters()])
class GCN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.validation_step_outputs =[]

        self.cnn = ModuleList()
        Layers_cnn =[1,32,32,32,64,64,64,128,128,128,256,256,256]
        n=0
        for in_channel, out_channel in zip(Layers_cnn, Layers_cnn[1:]):
            self.cnn.append(Conv2d(in_channel, out_channel, kernel_size=(1,conv2D_filter_size), padding='same'))
            self.cnn.append(ReLU())
            n = n+1
            if n ==3:
                self.cnn.append(AvgPool2d(kernel_size=(1, pooling_size)))
                #self.cnn.append(Dropout(0.2))
                n = 0
        
        
        self.bn1 = BatchNorm2d(256, eps = 0.001, momentum = 0.99, affine=True) # change the parameter to be same as using Keras
  
        self.gnn_1 = EdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(520*2, 520),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(520, 512),
                torch.nn.LeakyReLU(),
            ),
            aggr="add",
            k=neighbors,
        )

        self.gnn_2 = EdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(512*2, 512),  #256
                torch.nn.LeakyReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.LeakyReLU(),
            ),
            aggr="add",
            k=neighbors,
        )

        self.gnn_3 = EdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(256 * 2, 256),  #128
                torch.nn.LeakyReLU(),
                torch.nn.Linear(256, 64),   #64
                torch.nn.LeakyReLU(),
            ),
            aggr="add",
            k=neighbors,
        )
        
        self.gnn_4 = EdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(64*2, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.LeakyReLU(),
            ),
            aggr="add",
            nb_neighbors=neighbors,
        )
        
        
        self.out1 = Linear(32,1)
      
    def forward(self, x):
       
        for (l, layer) in zip(range(len(self.cnn)), self.cnn):
            x = layer(x)

        x = self.bn1(x)
        
        x = torch.transpose(x, 1, 2)
        x = torch.flatten(x, 2, 3) ## x.view(-1,2560)
        
        # pos has the position of each antenna [x,y,z]
        # antenna has the antenna features: [orientation_phi,orientation_theta,rotation_phy,rotation_theta,pos_x,pos_y,pos_z,type]
        pos = torch.tensor([[3,0,-3],[0,3,-3],[-3,0,-3],[0,-3,-3],[0,0,-15]])
        antenna = torch.tensor([[0,180,180,90,3,0,-3,0], [0,180,270,90,0,3,-3,0], [1,180,0,90,-3,0,-3,0], [0,180,90,90,0,-3,-3,0], [0,180,0,90,0,0,-15,1]], device=device)
        antenna_feat = antenna.unsqueeze(0).expand(x.size(dim=0), 5, 8)
        edge_index = knn_graph(pos, k=neighbors).to(device)
        x = torch.cat((x, antenna_feat), dim=2)
             
        x = self.gnn_1(x, edge_index)
        x = self.gnn_2(x, edge_index)
        x = self.gnn_3(x, edge_index)
        x = self.gnn_4(x, edge_index)

        x = torch.mean(x, 1, True)

        x = torch.flatten(x, 1)

        ener = self.out1(x)

        return ener

    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-7)  # change the parameter to be same as using Keras
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=es_patience-4, min_lr=0.000001, verbose=1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        #y = y.squeeze()
        pred_e = self.forward(x)
        loss = criterion(pred_e, y)
        # to record something
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=len(train_batch),  logger=True)#on_step=True,
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        #y = y.squeeze()
        pred_e = self.forward(x)
        loss = criterion(pred_e, y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=len(val_batch), logger=True)
        return loss
    
   


# model
model = GCN().to(device)
model.float
#model.cuda()

#print('dsss', count_params(model))

from torchsummary import summary
summary(model, (1,5,512))

#callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
#from pytorch_lightning.accelerators import find_usable_cuda_devices


mc = ModelCheckpoint(dirpath=saved_model_dir, filename= "latest_model_checkpoint", 
    monitor='val_loss', verbose=1)

es = EarlyStopping("val_loss", patience=es_patience, min_delta=es_min_delta, verbose=1)
callbacks = [es, mc] # DeviceStatsMonitor()

# Configuring CSV-logger : save epoch and loss values
from pytorch_lightning.loggers import CSVLogger
csv_logger = CSVLogger(saved_model_dir, version=1, flush_logs_every_n_steps=64)#, append=True)

#AVAIL_GPUS = min(1, torch.cuda.device_count())



# define training parameters
trainer = pl.Trainer(
    gpus=1 if torch.cuda.is_available() else None, 
    auto_select_gpus=True,  
    callbacks = callbacks, 
    max_epochs = epochs,
    logger = csv_logger,
    log_every_n_steps=200,
    precision = 32
    )

trainer.fit(model, dataloader_train, dataloader_val)

save_model_path=os.path.join(saved_model_dir, f"{run_name}.pt")
torch.save(model.state_dict(), save_model_path)

