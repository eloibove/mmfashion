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
from sklearn.neighbors import NearestNeighbors
import numpy as np
import csv

# Load the config from the custom file
cfg_fname = 'configs/retriever_in_shop/global_retriever_vgg_loss_id_triplet.py' # Triplet network
# cfg_fname = 'configs/retriever_in_shop/global_retriever_vgg_loss_id.py' # Plain siamese

cfg = Config.fromfile(cfg_fname)

# Data loader
#train_set = build_dataset(cfg.data.train)
query_set = build_dataset(cfg.data.query)
gallery_set = build_dataset(cfg.data.gallery)
print('datasets loaded',flush=True)


#train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.data.imgs_per_gpu, shuffle=True)
query_loader = torch.utils.data.DataLoader(query_set, batch_size=1, shuffle=False)
gallery_loader = torch.utils.data.DataLoader(gallery_set, batch_size=1, shuffle=False)

print('dataloader built',flush=True)

# Build model and load checkpoint
# model = build_retriever(cfg.model)
model = Vgg16L2(num_dim=128) 
model.load_state_dict(torch.load('checkpoint/siamese_hnm_100_epochs_8_12.pth'))
model.eval()
model.cuda()
cuda = True

"""
backbone_model = Vgg16L2(num_dim=128)
model_full = TripletNet(backbone_model)
model_full.load_state_dict(torch.load('checkpoint/triplet_semi_hn_100_epochs_8_8.pth'))
model = model_full.embedding_net
model.eval()
model.cuda()
cuda = True
"""

# Compute the embeddings for the gallery set
embeddings_gallery = []
ids_gallery = []
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(gallery_loader):
        #data = tuple(d.cuda() for d in data)
        data = data.cuda()
        outputs = model(data)
        embeddings_gallery.append(outputs[0].cpu().numpy())
        ids_gallery.append(target.item())

    print('embeddings computed!', flush=True)

    with open('embeddings.tsv', 'w') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        
        for i, row in enumerate(embeddings_gallery):
            writer.writerow(row)

    print('saved to file!', flush=True)