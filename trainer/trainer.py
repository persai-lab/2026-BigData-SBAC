import logging
import numpy as np
from sklearn import metrics
from sklearn.metrics import average_precision_score
from torch.backends import cudnn
import torch
from torch import nn
import warnings
import os 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from dataloader.dataloader import SBAC_DataLoader
from model.SBAC import SBAC

torch.cuda.empty_cache()
warnings.filterwarnings("ignore")

cudnn.benchmark = False 
cudnn.deterministic = True

class trainer(object):
    def __init__(self, config, data):
        self.config = config
        self.logger = logging.getLogger("trainer")
        self.mode = config.mode
        self.manual_seed = config.seed

        self.target_domain = getattr(config, 'target_domain', 'both').lower()
        self.aggregation_mode = getattr(config, 'aggregation_mode', 'both')

        self.model_name = getattr(config, 'model_name', 'SBAC')
        self.model = SBAC(config)
        
        self.data_loader = SBAC_DataLoader(config, data)

        self.num_items_q = config.num_items_q
        self.num_items_total = config.num_items

        self.current_epoch = 1
        self.current_iteration = 1
        self.early_stopping_limit = self.config.early_stopping_limit
        self.patience = 0

        self.best_val_ndcg = 0.
        self.best_val_prauc_buy = 0.
        self.best_epoch = 0 
        
        self.train_loss_list = []
        self.test_loss_list = []

        print(f">> Trainer Loaded. Model: {self.model_name} | Target Domain: {self.target_domain.upper()} <<")

        self.criterion_Item = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
        
        if config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                             lr=self.config.learning_rate,
                                             momentum=self.config.momentum,
                                             weight_decay=self.config.weight_decay)
        elif config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                              lr=self.config.learning_rate,
                                              betas=(config.beta1, config.beta2),
                                              eps=self.config.epsilon,
                                              weight_decay=self.config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='min',
            patience=0,
            min_lr=1e-5,
            factor=0.5,
        )

        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            torch.manual_seed(self.manual_seed)
            if self.config.gpu_device in range(torch.cuda.device_count()): 
                self.cuda_device = f"cuda:{self.config.gpu_device}"
            else:
                self.config.gpu_device = 0
                self.cuda_device = f"cuda:{self.config.gpu_device}"
            self.device = torch.device(f"{self.cuda_device}")
            self.model = self.model.to(self.device)
            
            # SEPARATE WEIGHTS FOR D1 and D2
            view_d1 = getattr(self.config, 'beh_wt_view_d1', self.config.beh_wt_view)
            buy_d1 = getattr(self.config, 'beh_wt_buy_d1', self.config.beh_wt_buy)
            cart_d1 = getattr(self.config, 'beh_wt_cart_d1', self.config.beh_wt_cart)
            weights_d1 = torch.tensor([view_d1, buy_d1, cart_d1]).to(self.device)
            self.criterion_CE_d1 = nn.CrossEntropyLoss(weight=weights_d1, reduction='sum').to(self.device)

            view_d2 = getattr(self.config, 'beh_wt_view_d2', self.config.beh_wt_view)
            buy_d2 = getattr(self.config, 'beh_wt_buy_d2', self.config.beh_wt_buy)
            cart_d2 = getattr(self.config, 'beh_wt_cart_d2', self.config.beh_wt_cart)
            weights_d2 = torch.tensor([view_d2, buy_d2, cart_d2]).to(self.device)
            self.criterion_CE_d2 = nn.CrossEntropyLoss(weight=weights_d2, reduction='sum').to(self.device)

            self.criterion_Item = self.criterion_Item.to(self.device)
            print(f"Beh weights D1: {weights_d1}")
            print(f"Beh weights D2: {weights_d2}")
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
        
        # LOAD MIXED DOMAIN ITEM POOLS
        self.d1_items_tensor = torch.from_numpy(self.data_loader.d1_items_array).long().to(self.device)
        self.d2_items_tensor = torch.from_numpy(self.data_loader.d2_items_array).long().to(self.device)
        self.d1_items_np = self.data_loader.d1_items_array
        self.d2_items_np = self.data_loader.d2_items_array

        # COMPUTE PER-USER TOTAL INTERACTION LENGTHS
        all_i1 = data["traindata"]["i1_data"] + data["testdata"]["i1_data"]
        all_i2 = data["traindata"]["i2_data"] + data["testdata"]["i2_data"]
        self.user_total_interactions = [
            sum(1 for x in i1 if x != 0) + sum(1 for x in i2 if x != 0)
            for i1, i2 in zip(all_i1, all_i2)
        ]

        self.best_model_path = "" 
    
    def train(self):
        agg_str  = "agg" if self.aggregation_mode != 'none' else "noagg"
        self.save_path_pattern = f"saved_model/{self.config.data_name}/{self.config.model_name}/{self.config.model_name}_{agg_str}_fold_{self.config.fold_cv}.pt"
        if self.config.mode == "train" and self.config.test_only == False:
            os.makedirs(os.path.dirname(self.save_path_pattern), exist_ok=True)

            for epoch in range(1, self.config.max_epoch + 1):
                if self.patience == self.early_stopping_limit:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
                
                print("="*50 + "Epoch {}".format(epoch) + "="*50)
                self.train_one_epoch()
                self.validate() 
                self.current_epoch += 1
            
            print(f"Best Epoch: {self.best_epoch} | Best Val {self.target_domain.upper()} PR-AUC (Buy): {self.best_val_prauc_buy:.4f} | NDCG@{self.config.rank_k}: {self.best_val_ndcg:.4f}")
            
            print("\n" + "="*50 + " FINAL TESTING " + "="*50)
            self.test()
            return self.best_val_ndcg, self.best_val_prauc_buy
        elif self.config.test_only:
            print("\n" + "="*50 + " FINAL TESTING " + "="*50)
            self.test()
    
    def train_one_epoch(self):
        self.model.train()
        self.train_loss = 0
        self.avg_train_loss = 0
        train_elements = 0 

        # TOWER LOSS WEIGHTS
        w_item_d1 = getattr(self.config, 'w_item_d1', 1.0)
        w_item_d2 = getattr(self.config, 'w_item_d2', 1.0)
        global_loss_weight = getattr(self.config, 'loss_weight', 0.1)
        w_beh_d1 = getattr(self.config, 'w_beh_d1', global_loss_weight)
        w_beh_d2 = getattr(self.config, 'w_beh_d2', global_loss_weight)

        for batch_idx, data in enumerate(self.data_loader.train_loader):
            i1_list, b_list, i2_list, d_list, target_answers_list, target_masks_list, target_masks_i2_list, raw_lens, agg_counts = data
            
            i1_list = i1_list.to(self.device)
            b_list = b_list.to(self.device)
            i2_list = i2_list.to(self.device)
            d_list = d_list.to(self.device)
            target_answers_list = target_answers_list.to(self.device)
            
            mask_d1_raw = target_masks_list.to(self.device)
            mask_d2_raw = target_masks_i2_list.to(self.device) 

            self.optimizer.zero_grad()
            
            # FORWARD
            out_beh_d1, out_beh_d2, out_item_d1, out_item_d2 = self.model(i1_list, b_list, i2_list, d_list) 

            # SLICING
            label = target_answers_list[:, 1:] 
            target_items_d1 = i1_list[:, 1:]
            target_items_d2 = i2_list[:, 1:]
            
            mask_d1 = (target_items_d1 != 0)
            mask_d2 = (target_items_d2 != 0)

            # BEHAVIOR LOSS
            binary_labels = label.long()
            loss_beh_d1 = 0.0
            loss_beh_d2 = 0.0
            
            if mask_d1.sum() > 0:
                pred_d1 = out_beh_d1[mask_d1]
                lbl_d1 = binary_labels[mask_d1]
                loss_beh_d1 = self.criterion_CE_d1(pred_d1, lbl_d1)

            if mask_d2.sum() > 0:
                pred_d2 = out_beh_d2[mask_d2]
                lbl_d2 = binary_labels[mask_d2]
                loss_beh_d2 = self.criterion_CE_d2(pred_d2, lbl_d2)

            # ITEM PREDICTION LOSS
            loss_item_d1 = 0.0
            loss_item_d2 = 0.0
            
            valid_item_mask_d1 = (label < 3) & mask_d1
            if valid_item_mask_d1.sum() > 0:
                masked_latents_d1 = out_item_d1[valid_item_mask_d1]
                flat_out_d1 = self.model.W_item_d1(masked_latents_d1)
                flat_lbl_d1 = target_items_d1[valid_item_mask_d1]
                loss_item_d1 = self.criterion_Item(flat_out_d1, flat_lbl_d1.long())

            valid_item_mask_d2 = (label < 3) & mask_d2
            if valid_item_mask_d2.sum() > 0:
                masked_latents_d2 = out_item_d2[valid_item_mask_d2]
                flat_out_d2 = self.model.W_item_d2(masked_latents_d2)
                flat_lbl_d2 = target_items_d2[valid_item_mask_d2]
                loss_item_d2 = self.criterion_Item(flat_out_d2, flat_lbl_d2.long())

            # TOTAL LOSS CALCULATION
            loss = (w_item_d1 * loss_item_d1) + (w_item_d2 * loss_item_d2) + \
                       (w_beh_d1 * loss_beh_d1) + (w_beh_d2 * loss_beh_d2)

            self.train_loss += loss.item()
            train_elements += 1 
            
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        self.avg_train_loss = self.train_loss / train_elements if train_elements > 0 else 0.0
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        self.scheduler.step(self.avg_train_loss)
        self.train_loss_list.append(self.avg_train_loss)
        
        print(f"Train Loss (Total): {self.train_loss:.6f} | Avg Loss: {self.avg_train_loss:.6f} | Batches: {train_elements}")
    
    def validate(self):
        self.run_evaluation(self.data_loader.val_loader, mode='val')

    def test(self):
        if os.path.exists(self.save_path_pattern):
            print(f"Loading best model from: {self.save_path_pattern}")
            self.model.load_state_dict(torch.load(self.save_path_pattern, map_location=self.device))
        self.run_evaluation(self.data_loader.test_loader, mode='test')

    def run_evaluation(self, loader, mode='val'):
        self.model.eval()
        eval_loss = 0
        eval_elements = 0
        
        w_item_d1 = getattr(self.config, 'w_item_d1', 1.0)
        w_item_d2 = getattr(self.config, 'w_item_d2', 1.0)
        global_loss_weight = getattr(self.config, 'loss_weight', 0.1)
        w_beh_d1 = getattr(self.config, 'w_beh_d1', global_loss_weight)
        w_beh_d2 = getattr(self.config, 'w_beh_d2', global_loss_weight)

        d1_metrics = {'beh_probs': [], 'beh_labels': [], 'ndcg': [], 'hr': []}
        d2_metrics = {'beh_probs': [], 'beh_labels': [], 'ndcg': [], 'hr': []}
        
        
        rng = np.random.RandomState(self.manual_seed)
        
        with torch.inference_mode():
            for batch_i, data in enumerate(loader):
                i1_list, b_list, i2_list, d_list, target_answers_list, _, _, raw_lens, agg_counts = data
                
                i1_list = i1_list.to(self.device)
                b_list = b_list.to(self.device)
                i2_list = i2_list.to(self.device)
                d_list = d_list.to(self.device)
                target_answers_list = target_answers_list.to(self.device)
                
                out_beh_d1, out_beh_d2, out_item_d1, out_item_d2 = self.model(i1_list, b_list, i2_list, d_list) 

                label = target_answers_list[:, 1:]
                target_items_d1 = i1_list[:, 1:]
                target_items_d2 = i2_list[:, 1:]
                mask_d1 = (target_items_d1 != 0)
                mask_d2 = (target_items_d2 != 0)
                binary_labels = label.long()
                
                loss_beh_d1, loss_beh_d2 = 0.0, 0.0
                if mask_d1.sum() > 0: loss_beh_d1 = self.criterion_CE_d1(out_beh_d1[mask_d1], binary_labels[mask_d1]).item()
                if mask_d2.sum() > 0: loss_beh_d2 = self.criterion_CE_d2(out_beh_d2[mask_d2], binary_labels[mask_d2]).item()

                loss_item_d1, loss_item_d2 = 0.0, 0.0
                valid_item_mask_d1 = (label < 3) & mask_d1
                if valid_item_mask_d1.sum() > 0:
                    loss_item_d1 = self.criterion_Item(self.model.W_item_d1(out_item_d1[valid_item_mask_d1]), target_items_d1[valid_item_mask_d1].long()).item()
                valid_item_mask_d2 = (label < 3) & mask_d2
                if valid_item_mask_d2.sum() > 0:
                    loss_item_d2 = self.criterion_Item(self.model.W_item_d2(out_item_d2[valid_item_mask_d2]), target_items_d2[valid_item_mask_d2].long()).item()

                batch_loss = (w_item_d1 * loss_item_d1) + (w_item_d2 * loss_item_d2) + \
                                 (w_beh_d1 * loss_beh_d1) + (w_beh_d2 * loss_beh_d2)

                eval_loss += batch_loss
                eval_elements += 1 

                batch_size = i1_list.size(0)
                
                for b in range(batch_size):
                    seq_raw_len = raw_lens[b].item()
                    seq_agg_count = agg_counts[b].item()
                    
                    indices_d1 = (target_items_d1[b] != 0).nonzero(as_tuple=True)[0]
                    for idx in indices_d1:
                        idx = idx.item()
                        true_beh = label[b, idx].item()
                        true_item_id = target_items_d1[b, idx].item()
                        
                        if true_beh == 1 or true_beh == 2:
                            binary_target = 1 if true_beh == 1 else 2
                            pred_probs = torch.softmax(out_beh_d1[b, idx], dim=0).tolist()
                            d1_metrics['beh_probs'].append(pred_probs)
                            d1_metrics['beh_labels'].append(binary_target)
                        
                        if true_beh == 1:
                            step_latent = out_item_d1[b, idx]
                            step_scores = self.model.W_item_d1(step_latent)
                            ndcg_val, hr_val = self._calc_ranking(step_scores, true_item_id, d1_metrics, rng, is_d1=True)

                    indices_d2 = (target_items_d2[b] != 0).nonzero(as_tuple=True)[0]
                    for idx in indices_d2:
                        idx = idx.item()
                        true_beh = label[b, idx].item()
                        true_item_id = target_items_d2[b, idx].item()
                        
                        if true_beh == 1 or true_beh == 2:
                            binary_target = 1 if true_beh == 1 else 2
                            pred_probs = torch.softmax(out_beh_d2[b, idx], dim=0).tolist()
                            d2_metrics['beh_probs'].append(pred_probs)
                            d2_metrics['beh_labels'].append(binary_target)
                        
                        if true_beh == 1:
                            step_latent = out_item_d2[b, idx]
                            step_scores = self.model.W_item_d2(step_latent)
                            ndcg_val, hr_val = self._calc_ranking(step_scores, true_item_id, d2_metrics, rng, is_d1=False)

        whole_metrics = {
            'beh_probs': d1_metrics['beh_probs'] + d2_metrics['beh_probs'],
            'beh_labels': d1_metrics['beh_labels'] + d2_metrics['beh_labels'],
            'ndcg': d1_metrics['ndcg'] + d2_metrics['ndcg'],
            'hr': d1_metrics['hr'] + d2_metrics['hr']
        }
        metrics_data = {'d1': d1_metrics, 'd2': d2_metrics, 'whole': whole_metrics}
        if eval_elements > 0: avg_eval_loss = eval_loss / eval_elements
        else: avg_eval_loss = 0.0
        
        if mode == 'val':
            print(f"{mode.upper()} Loss (Total): {eval_loss:.6f} | Avg Loss: {avg_eval_loss:.6f} | Batches: {eval_elements}")
            self.track_best(metrics_data)
        elif mode == 'test':
            self.print_full_metrics(metrics_data)

    def _calc_ranking(self, step_scores, true_item_id, metrics_dict, rng, is_d1):
        true_score = step_scores[true_item_id].item()
        negatives = []
        
        target_pool = self.d1_items_np if is_d1 else self.d2_items_np
        if len(target_pool) == 0: target_pool = np.arange(1, self.num_items_total + 1)
        while len(negatives) < 99:
            sample_id = rng.choice(target_pool)
            if sample_id != true_item_id: negatives.append(sample_id)
        
        negatives_tensor = torch.tensor(negatives).long().to(self.device)
        neg_scores = step_scores[negatives_tensor]
        count_bigger = (neg_scores > true_score).sum().item()
        rank = count_bigger + 1
        
        ndcg_val = 1.0 / np.log2(rank + 1) if rank <= 10 else 0.0
        hr_val = 1.0 if rank <= 10 else 0.0
        
        metrics_dict['ndcg'].append(ndcg_val)
        metrics_dict['hr'].append(hr_val)

        return ndcg_val, hr_val 

    def track_best(self, metrics_data):
        target_data = metrics_data[self.target_domain]
        beh_labels = np.array(target_data['beh_labels'])
        beh_probs = np.array(target_data['beh_probs'])
        ndcg_list = target_data['ndcg']
        
        pr_auc_score_buy = 0.0
        if len(beh_labels) > 0 and len(np.unique(beh_labels)) >= 2:
            try:
                pr_auc_score_buy = average_precision_score((beh_labels == 1).astype(int), beh_probs[:, 1])
            except ValueError: pass

        ndcg_val = np.mean(ndcg_list) if ndcg_list else 0.0
        current_metric = ndcg_val + pr_auc_score_buy
        
        print(f"Validation ({self.target_domain.upper()}): PR-AUC Buy: {pr_auc_score_buy:.4f} | NDCG@10: {ndcg_val:.4f}")

        if current_metric > self.best_val_ndcg + self.best_val_prauc_buy: 
            self.best_val_ndcg = ndcg_val
            self.best_val_prauc_buy = pr_auc_score_buy
            self.best_epoch = self.current_epoch
            if hasattr(self, 'save_path_pattern'):
                torch.save(self.model.state_dict(), self.save_path_pattern)
                print(f"*** New Best Model (Composite Score: {current_metric:.4f}) ***")
            self.patience = 0
        else:
            self.patience += 1
            print(f"No improvement. Patience: {self.patience}")

    def print_full_metrics(self, metrics_data):
        print("\n" + "="*20 + " FINAL TEST RESULTS " + "="*20)
        for domain in ['d1', 'd2', 'whole']:
            data = metrics_data[domain]
            beh_labels = np.array(data['beh_labels'])
            beh_probs = np.array(data['beh_probs'])
            ndcg_list = data['ndcg']
            hr_list = data['hr']
            
            pr_auc_buy = 0.0
            pr_auc_cart = 0.0
            if len(beh_labels) > 0 and len(np.unique(beh_labels)) >= 2:
                try:
                    target_buy = (beh_labels == 1).astype(int)
                    target_cart = (beh_labels == 2).astype(int)
                    pr_auc_buy = average_precision_score(target_buy, beh_probs[:, 1])
                    pr_auc_cart = average_precision_score(target_cart, beh_probs[:, 2])
                except ValueError: pass
            
            print(f"\n--- {domain.upper()} ---")
            print(f"PR-AUC (Buy):  {pr_auc_buy:.4f}")
            print(f"PR-AUC (Cart): {pr_auc_cart:.4f}")
            print(f"NDCG@10:       {np.mean(ndcg_list):.4f}")
            print(f"HR@10:         {np.mean(hr_list):.4f}")
            print(f"Samples (Beh): {len(beh_labels)} | Samples (Rank): {len(ndcg_list)}")
        print("="*60 + "\n")
