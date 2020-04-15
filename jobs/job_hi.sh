#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 1000 # 2GB solicitados.
#SBATCH -p mhigh,mhigh # or mlow Partition to submit to master low prioriy queue
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o jobs/%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e jobs/%x_%u_%j.err # File to which STDERR will be written
python tools/train_in_shop.py -p mhigh
#python -m detectron2.utils.collect_env