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
from networks import TripletNet, Vgg16L2
from losses import TripletLoss
from trainer import fit

# Load the config from the custom file
cfg_fname = 'configs/retriever_in_shop/global_retriever_vgg_loss_id_triplet.py' # Triplet network
# cfg_fname = 'configs/retriever_in_shop/global_retriever_vgg_loss_id.py' # Plain siamese

cfg = Config.fromfile(cfg_fname)

# Data loader
train_set = build_dataset(cfg.data.train)
query_set = build_dataset(cfg.data.query)
gallery_set = build_dataset(cfg.data.gallery)
print('datasets loaded')


train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.data.imgs_per_gpu, shuffle=True)
query_loader = torch.utils.data.DataLoader(query_set, batch_size=cfg.data.imgs_per_gpu, shuffle=True)

"""
train_loader = build_dataloader(
                train_set,
                cfg.data.imgs_per_gpu,
                cfg.data.workers_per_gpu,
                len(cfg.gpus.train),
                dist=False)
query_loader = build_dataloader(
                query_set,
                cfg.data.imgs_per_gpu,
                cfg.data.workers_per_gpu,
                len(cfg.gpus.train),
                dist=False)
"""

print('dataloader built')



# Build model and load checkpoint
# model = build_retriever(cfg.model)
backbone_model = Vgg16L2(num_dim=128)

margin = 1.
model = TripletNet(backbone_model)
model.cuda()
cuda = True
print('model built')

loss_fn = TripletLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 500


print('start training')
fit(train_loader, query_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
print('done training')

torch.save(model.state_dict(), './checkpoint/trained_model_20_epochs.pth')

"""
# Test
test_retriever(
    model,
    query_set,
    gallery_set,
    cfg,
    distributed=False)
print('done testing')
"""