from easydict import EasyDict
import argparse
import pickle
from trainer.trainer import trainer
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import os

def seed_everything(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def single_exp(config):
    seed_everything(config.get('seed', 1024))

    config = EasyDict(config)

    parser = argparse.ArgumentParser(description='Run SBAC model experiments.')
    parser.add_argument('--lr', '--learning_rate', type=float, help='Set the learning rate.')
    parser.add_argument('--fold', '--fold_cv', type=float, help='Set the fold number.')

    args = parser.parse_args()

    if args.lr is not None:
        config.learning_rate = args.lr

    if args.fold is not None:
        config.fold_cv = int(args.fold)

    print(f"Config: {config}")

    data = pickle.load(open(f'data/{config.data_name}/train_test_1.pkl', 'rb'))
    config.num_items_q = data["num_items_Q"]
    config.num_items = data["num_items_Q"] + data["num_items_L"]
    config.num_users = data["num_users"]
    config.num_domains = 2

    print(f'Number of items in domain 1: {data["num_items_Q"]}')
    print(f'Number of items in domain 2: {data["num_items_L"]}')
    print(f"Total number of items: {config.num_items}")
    print(f"Number of users: {config.num_users}")
    print(f"Number of domains: {config.num_domains}")

    exp_trainer = trainer(config, data)
    exp_trainer.train()

def taobao():
    config = {
        "data_name": "taobao",
        "model_name": "SBAC",
        "mode": "train",
        "test_only": False,
        "shuffle": True,
        "num_folds": 5,
        "fold_cv": 1,

        "cuda": True,
        "gpu_device": 0,
        "seed": 1024,

        "aggregation_mode": "both",

        # Domain 1 Clustering Parameters
        "n_clusters_d1": 15,
        "window_size_d1": 15,
        "w2v_epoch_d1": 10,
        
        # Domain 2 Clustering Parameters
        "n_clusters_d2": 9,
        "window_size_d2": 20,
        "w2v_epoch_d2": 15,
        
        "learning_rate": 2.2107920275725755e-05,
        "loss_weight": 1.6,
        
        "w_item_d1": 1.0,
        "w_item_d2": 1.2,
        "w_beh_d1": 1.5,
        "w_beh_d2": 0.8,
        
        "dropout_rate": 0.1,
        "hidden_size": 64, # hidden state size

        "num_blocks": 1, 
        "num_heads": 4,
        "beh_wt_view": 0.0,
        "beh_wt_buy": 0.4,
        "beh_wt_cart": 1.0,
        
        "beh_wt_view_d1": 0.02,
        "beh_wt_buy_d1": 3.1,
        "beh_wt_cart_d1": 2.9,
        "beh_wt_view_d2": 0.03,
        "beh_wt_buy_d2": 4.3,
        "beh_wt_cart_d2": 4.3,
        
        "max_seq_len": 50,
        "max_seq_len_train": 50,
        "batch_size": 32,
        "max_epoch": 100,
        "validation_split": 0.2,
        "early_stopping_limit": 5,
        "target_domain": "whole", # ('d1', 'd2', 'whole') Target domain for early stopping criteria

        "embedding_size_item": 64, # item embedding size
        "embedding_size_b": 64, # behavior embedding size
        "embedding_size_domain": 32,

        "rank_k": 10,

        "init_std": 0.2,
        "max_grad_norm": 5,

        "optimizer": "adam",
        "epsilon": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.0001,
    }
    single_exp(config)


def ijcai():
    config = {
        "data_name": "ijcai",
        "model_name": "SBAC",
        "mode": "train",
        "test_only": False,
        "shuffle": True,
        "num_folds": 5,
        "fold_cv": 1,

        "cuda": True,
        "gpu_device": 0,
        "seed": 1024,

        "aggregation_mode": "both",

        # Domain 1 Clustering Parameters
        "n_clusters_d1": 30,
        "window_size_d1": 10,
        "w2v_epoch_d1": 20,
        
        # Domain 2 Clustering Parameters
        "n_clusters_d2": 5,
        "window_size_d2": 20,
        "w2v_epoch_d2": 20,

        "learning_rate" : 0.000349235874556232,
        "loss_weight" : 1.8,

        "w_item_d1": 1.0,
        "w_item_d2": 1.0,
        "w_beh_d1": 1.8,
        "w_beh_d2": 1.7,

        "dropout_rate" : 0.1,
        "hidden_size" : 128, # hidden state size
        
        "num_blocks" : 3,
        "num_heads" : 2,  
        "beh_wt_view": 0.0,
        "beh_wt_buy": 1.3,
        "beh_wt_cart": 1.4,

        "beh_wt_view_d1": 0.00001,
        "beh_wt_buy_d1": 4.8,
        "beh_wt_cart_d1": 3.0,
        "beh_wt_view_d2": 0.00003,
        "beh_wt_buy_d2": 4.0,
        "beh_wt_cart_d2": 1.5,
        
        "max_seq_len": 50,
        "max_seq_len_train": 50,
        "batch_size": 32,
        "max_epoch": 100,
        "validation_split": 0.2,
        "early_stopping_limit": 5,
        "target_domain": "whole", # ('d1', 'd2', 'whole') Target domain for early stopping criteria

        "embedding_size_item": 64, # item embedding size
        "embedding_size_b": 64, # behavior embedding size
        "embedding_size_domain": 32,
        
        "rank_k": 10,

        "init_std": 0.2,
        "max_grad_norm": 5,

        "optimizer": "adam",
        "epsilon": 0.1,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.01,
    }
    single_exp(config)


if __name__ == "__main__":
    taobao()
    # ijcai()