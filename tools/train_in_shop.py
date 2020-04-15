from __future__ import division
import argparse

from mmcv import Config
from mmfashion.apis import (get_root_logger, init_dist, set_random_seed,
                            train_retriever, test_retriever)
from mmfashion.datasets import build_dataset
from mmfashion.models import build_retriever
from mmfashion.utils import init_weights_from


# Load the config from the custom file
cfg_fname = 'configs/retriever_in_shop/global_retriever_vgg_loss_id_triplet.py' #'configs/retriever_in_shop/global_retriever_resnet.py'
cfg = Config.fromfile(cfg_fname)

# Data loader
train_set = build_dataset(cfg.data.train)
query_set = build_dataset(cfg.data.query)
gallery_set = build_dataset(cfg.data.gallery)
print('datasets loaded')

# Build model and load checkpoint
model = build_retriever(cfg.model)
print('model built')


# Train
train_retriever(
    model,
    train_set,
    cfg,
    distributed=False)

# Test
test_retriever(
    model,
    query_set,
    gallery_set,
    cfg,
    distributed=False)