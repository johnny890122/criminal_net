import pandas as pd
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import random
import networkx as nx

from GraphEmbedding.ge import DeepWalk
from utility import show_arg
import json
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    parser = ArgumentParser()
    # random seed
    parser.add_argument("--seed", type=int, default=0)
    # directory path
    parser.add_argument("--data_dir", type=Path, default="./data/")
    parser.add_argument("--result_dir", type=Path, default="./result/")
    # deep wallk parameter
    parser.add_argument("--walk_length", type=int, default=10)
    parser.add_argument("--num_walks", type=int, default=80)
    parser.add_argument("--workers", type=int, default=48)

    parser.add_argument("--embed_size", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--iter", type=int, default=3)

    # ouput file
    parser.add_argument("--output_file", type=str, default="train_embedding.csv")
    args = parser.parse_args()
    return args

def main(args):

    TRAIN = "train"
    DEV = "dev"
    SPLITS = [TRAIN, DEV]

    print("Loading data...\n")
    data_paths = {split: args.data_dir/"{}.edgelist".format(split) for split in SPLITS}
    data = {
        split: nx.read_edgelist(str(path), create_using=nx.Graph(), nodetype=str, data=[('weight',int)])
            for split, path in data_paths.items() }

    print("Loading model and start sampling...")
    sampling_start_time = time.time()
    model = DeepWalk(data[TRAIN],
                     walk_length=args.walk_length, num_walks=args.num_walks, workers=args.workers,
    )
    sampling_end_time = time.time()
    print("Spend {:.2f} seconds...\n".format(sampling_end_time - sampling_start_time))

    print("Start training...")
    tr_start_time = time.time()
    model.train(embed_size=args.embed_size, window_size=args.window_size, iter=args.iter)# train model
    tr_end_time = time.time()
    print("Spend {:.2f} seconds...\n".format(tr_end_time - tr_start_time))

    print("Save embedding...")
    embeddings = model.get_embeddings()

    df = pd.DataFrame.from_dict(embeddings, orient='index')
    df.to_csv(args.result_dir/args.output_file)

if __name__ == "__main__":
    args = parse_args()
    show_arg(args)
    main(args)

