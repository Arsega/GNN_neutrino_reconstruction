# Imports
import os
import numpy as np

import time
from toolbox import load_file
from constants import datapath, data_filename, label_filename, n_files, n_files_val

import torch
import torch.nn as nn
from torch.nn import Dropout, ReLU, ModuleList, MSELoss, AvgPool2d, Conv2d, BatchNorm2d
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch_geometric.data import Data
import pytorch_lightning as pl 
from torch.nn import Dropout, ReLU, ModuleList, MSELoss
from torch_geometric.nn import GCNConv, TopKPooling, BatchNorm, Linear, GraphNorm, EdgePooling, EdgeConv, DynamicEdgeConv, global_mean_pool, avg_pool_neighbor_x, avg_pool_x
from torch_geometric.nn.pool import knn_graph
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum


# -------
device = torch.device("cuda")

np.set_printoptions(precision=4)

# n_files and n_files_val comes from dataset in constants.py
n_files_test = 1
norm = 1e-6

n_files_train = n_files - n_files_val - n_files_test
list_of_file_ids_train = np.arange(n_files_train, dtype=np.int)
list_of_file_ids_val = np.arange(n_files_train, n_files_train + n_files_val, dtype=np.int)
list_of_file_ids_test = np.arange(n_files_train + n_files_val, n_files, dtype=np.int)
n_events_per_file = 100000
batch_size = 64


#dataset
class Prepare_Dataset(Dataset):
    def __init__(self, file_ids, points=99000, transform=None, target_transform=None):
        self.file_ids = file_ids
        self.data = None
        self.labels = None
        self.points = points

    def __len__(self):
        return len(self.file_ids)*self.points

    def __getitem__(self, idx):
        #print('time0: ', time.time())
        file_idx = idx // self.points
        if np.mod((idx+self.points),self.points) == 0:
            self.data, self.labels = self.get_data(file_idx)
        #print('time1: ', time.time())
        return self.data[idx-self.points*file_idx].to(torch.float32), self.labels[idx-self.points*file_idx].to(torch.float32)

    def get_data(self,file_idx):
        # Load data from file
        data, labels = load_file(self.file_ids[file_idx],norm)
        # randomly choose the points in a file 
        idxx = np.random.choice(data.shape[0], size=len(data), replace=False)
        # swap the axes since inputs shape in torch is (batch, channel, input dimension1, input dimension2)
        data = np.swapaxes(data,1,3)
        data = np.swapaxes(data,2,3)
        labels = np.expand_dims(labels,1)
        data = data[idxx,:]
        labels = labels[idxx,:]
        data = data[0:self.points]
        labels = labels[0:self.points]
        data = torch.from_numpy(data)
        labels = torch.from_numpy(labels)
        return data, labels


class Prepare_Dataset_val(Dataset):
    def __init__(self, file_ids, points=64, transform=None, target_transform=None):
        # transform and target_transform are useless in our case
        self.file_ids = file_ids
        self.data = None
        self.labels = None
        self.transform = transform
        self.target_transform = target_transform
        #self.file_idx = None
    def __len__(self):
        return len(self.file_ids)*99000

    def __getitem__(self, idx):
        #print(idx)
        file_idx = idx // 99000
        #print(file_idx)
        #print( 'idx ', idx, 'file_idx ', self.file_idx)
        if np.mod(idx+99000,99000) == 0:
            self.data, self.labels = self.get_data(file_idx)
        return self.data[idx-99000*file_idx], self.labels[idx-99000*file_idx]
       

    def get_data(self,file_idx):
        # Load data from file
        data, shower_energy_log10 = load_file(self.file_ids[file_idx],norm)
        print('loading',self.file_ids[file_idx] )
        # randomly choose the points in a file 
        idxx = np.random.choice(shower_energy_log10.shape[0], size=len(shower_energy_log10), replace=False)
        # swap the axes since inputs shape in torch is (batch, channel, input dimension1, input dimension2)
        data = np.swapaxes(data,1,3)
        data = np.swapaxes(data,2,3)

        #data = np.swapaxes(data,1,2)

        shower_energy_log10 = np.expand_dims(shower_energy_log10,1)

        data = data[idxx,:]
        shower_energy_log10 = shower_energy_log10[idxx,:]
        data = data[:99000]
        shower_energy_log10 = shower_energy_log10[:99000]
        print('nnnn')
        print(data.shape)
        print(shower_energy_log10.shape)

        data = torch.from_numpy(data)
        labels = torch.from_numpy(shower_energy_log10)
        return data, labels




#model for importing
conv2D_filter_size = 5
pooling_size = 4
amount_Conv2D_layers_per_block = 3 
amount_Conv2D_blocks = 4
conv2D_filter_amount = 32
neighbors = 4

class GCN(pl.LightningModule):
    def __init__(self):
        super().__init__()

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
                torch.nn.Linear(520, 1024),
                torch.nn.LeakyReLU(),
            ),
            aggr="add",
            nb_neighbors=neighbors,
        )

        self.gnn_2 = EdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(1024*2, 1024),  #256
                torch.nn.LeakyReLU(),
                torch.nn.Linear(1024, 512),
                torch.nn.LeakyReLU(),
            ),
            aggr="add",
            nb_neighbors=neighbors,
        )

        self.gnn_3 = EdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(512 * 2, 512),  #128
                torch.nn.LeakyReLU(),
                torch.nn.Linear(512, 256),   #64
                torch.nn.LeakyReLU(),
            ),
            aggr="add",
            nb_neighbors=neighbors,
        )
        
        self.gnn_4 = EdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(256*2, 256),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(256, 64),
                torch.nn.LeakyReLU(),
            ),
            aggr="add",
            nb_neighbors=neighbors,
        )
        
        
        self.out1 = Linear(320,3)
       
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

        x = torch.flatten(x, 1)

        ener = self.out1(x)
        return ener

