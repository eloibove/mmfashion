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
from networks import TripletNet, Vgg16L2, EmbeddingNet, SiameseNet
from losses import TripletLoss, ContrastiveLoss
from trainer import fit

# Load the config from the custom file
cfg_fname = 'configs/retriever_in_shop/triplet_hnm.py' # Triplet network with hard negative mining
# cfg_fname = 'configs/retriever_in_shop/triplet_vanilla.py' # Triplet network 

cfg = Config.fromfile(cfg_fname)
cuda = True


from datasets import BalancedBatchSampler
# Datasets
train_set = build_dataset(cfg.data.train)
query_set = build_dataset(cfg.data.query)
gallery_set = build_dataset(cfg.data.gallery)
print('datasets loaded')

# Mini batch selector
train_batch_sampler = BalancedBatchSampler(torch.tensor(train_set.train_labels), n_classes=8, n_samples=12)
test_batch_sampler = BalancedBatchSampler(torch.tensor(query_set.train_labels), n_classes=8, n_samples=12)

# Dataloaders
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(train_set, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(query_set, batch_sampler=test_batch_sampler, **kwargs)
print('dataloaders built')


# Build model and load checkpoint
# model = build_retriever(cfg.model)
model = Vgg16L2(num_dim=128)
model.cuda()
print('model built')

# Set up the network and training parameters
from losses import OnlineContrastiveLoss
from utils import AllPositivePairSelector, HardNegativePairSelector

margin = 1.
loss_fn = OnlineContrastiveLoss(margin, AllPositivePairSelector())
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 100
log_interval = 50

print('start training')
fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, model_name='siamese_app_8_12')
print('done training')

torch.save(model.state_dict(), './checkpoint/siamese_app_100_epochs_8_12.pth')
