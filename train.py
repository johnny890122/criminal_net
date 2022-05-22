#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import random
import networkx as nx

from GraphEmbedding.ge import DeepWalk 


# In[24]:


get_ipython().system('pip install gensim')


# In[18]:


def parse_args():
    parser = ArgumentParser()
     
    # random seed
    parser.add_argument("--seed", type=int, default=0)
    
    # directory path
    parser.add_argument("--data_dir", type=Path, default="./data/")
    
    
    
    # deep wallk parameter
    parser.add_argument("--walk_length", type=int, default=10)
    parser.add_argument("--num_walks", type=int, default=80)
    parser.add_argument("--workers", type=int, default=1)
    
    parser.add_argument("--embed_size", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--iter", type=int, default=3)

    args = parser.parse_args(args=[])
    return args

# walk length t -> walk_length (Length of the random walk started at each node)
# walks per vertex γ -> num_walks(Number of random walks to start at each node)

# window size w  -> window_size (Window size of skipgram model)
# embedding size d -> embed_size (representation-size)


# In[19]:


def main(args):
    
    TRAIN = "train"
    DEV = "dev"
    SPLITS = [TRAIN, DEV]

    data_paths = {split: args.data_dir/"{}.edgelist".format(split) for split in SPLITS}
    data = {
        split: nx.read_edgelist(path, create_using=nx.Graph(), nodetype=int, data=[('weight',int)])
            for split, path in data_paths.items() }
    
    model = DeepWalk(data[TRAIN], 
                     walk_length=args.walk_length, num_walks=args.num_walks, workers=args.workers,
    )
    model.train(embed_size=args.embed_size, window_size=args.window_size, iter=args.iter)# train model
    
    embeddings = model.get_embeddings()
    return embeddings
    
args = parse_args()
tmp = main(args)


# In[ ]:





# In[ ]:


G = # Read graph



model = DeepWalk(G, walk_length=10, num_walks=80,workers=1)#init model
model.train(window_size=5,iter=3)# train model
embeddings = model.get_embeddings()# get embedding vectors

