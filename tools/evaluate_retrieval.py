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
model.load_state_dict(torch.load('checkpoint/siamese_app_100_epochs_8_12.pth'))
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

    # Count the number of occurences of each query
    unique, counts = np.unique(ids_gallery, return_counts=True)
    occurrences = dict(zip(unique, counts))

    # For each query, compute the descriptor and find the K-NN
    K=[3, 5, 10, 20]
    knn = NearestNeighbors(n_neighbors=max(K), algorithm='brute')
    knn.fit(embeddings_gallery)

    # Precompute the neighbors using the highest K
    neighbor_list = []
    ids_query = []
    for batch_idx, (data, target) in enumerate(query_loader):
        data = data.cuda()
        outputs = model(data)
        query_embedding = outputs[0].cpu().numpy()
        query_id = target.item()
        ids_query.append(query_id)
        neighbors = knn.kneighbors([query_embedding], max(K), return_distance=False)
        neighbor_list.append(neighbors)

    print('Computed descriptors! Evaluating...',flush=True)

    for k in K:
        recall = 0
        for idx, neighbors in enumerate(neighbor_list):
            query_id = ids_query[idx]
            counts = 0
            for i, n in enumerate(neighbors[0]):
                if(i > k):
                    break
                if(ids_gallery[n]==query_id):
                    counts += 1
            recall += counts/min(k,occurrences[query_id])
        recall /= len(query_loader)
        print('Recall @ ' + str(k) + ' = ' + str(recall),flush=True)

