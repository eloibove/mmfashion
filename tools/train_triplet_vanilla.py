from __future__ import division
import argparse

from mmcv import Config
from mmfashion.apis import (get_root_logger, init_dist, set_random_seed,
                            train_retriever, test_retriever)
from mmfashion.datasets import build_dataset, build_dataloader
from mmfashion.models import build_retriever
from mmfashion.utils import init_weights_from

from mmfashion.models import builder

import torch
from torch.optim import lr_scheduler
import torch.optim as optim

import sys
sys.path.insert(0, "/home/grupo08/M5/MetricLearning/bielski")
from networks import TripletNet, Vgg16L2, EmbeddingNet
from losses import TripletLoss
from trainer import fit

# Load the config from the custom file
# cfg_fname = 'configs/retriever_in_shop/triplet_hnm.py' # Triplet network with hard negative mining
cfg_fname = 'configs/retriever_in_shop/triplet_vanilla.py' # Triplet network 

cfg = Config.fromfile(cfg_fname)
cuda = True


# Data loader
train_set = build_dataset(cfg.data.train)
query_set = build_dataset(cfg.data.query)
print('datasets loaded')

train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.data.imgs_per_gpu, shuffle=True)
query_loader = torch.utils.data.DataLoader(query_set, batch_size=cfg.data.imgs_per_gpu, shuffle=True)
print('dataloader created')

# Build model
backbone_model = Vgg16L2(num_dim=128)
model = TripletNet(backbone_model)
model.cuda()
print('model built')

margin = 1.
loss_fn = TripletLoss(margin)
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 500

print('start training')
fit(train_loader, query_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
print('done training')

torch.save(model.state_dict(), './checkpoint/triplet_vanilla_20_epochs_1e-4.pth')