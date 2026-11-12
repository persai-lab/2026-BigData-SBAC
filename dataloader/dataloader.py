import numpy as np
import torch
from sklearn.model_selection import train_test_split, KFold
import os
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import TensorDataset
import more_itertools as miter
from build_clusters import Build_Clusters

class SBAC_DataLoader:
    def __init__(self, config, data):
        self.config = config
        self.batch_size = config.batch_size
        self.shuffle = config.shuffle
        self.collate_fn = default_collate

        self.seed = config.seed
        self.validation_split = config.validation_split
        self.mode = config.mode
        
        # Cross-validation Specific Configs
        self.fold_cv = getattr(config, 'fold_cv', 1)  # 1-based index (e.g., 1, 2, 3, 4, 5)
        self.num_folds = getattr(config, 'num_folds', 5)
        
        # Aggregation config
        self.aggregation_mode = getattr(config, 'aggregation_mode', 'both')

        self.min_seq_len = config.min_seq_len if "min_seq_len" in config else None
        self.max_seq_len = config.max_seq_len if "max_seq_len" in config else None
        self.max_seq_len_train = config.max_seq_len_train if "max_seq_len_train" in config else None
        self.stride = config.max_seq_len if "max_seq_len" in config else None
        self.stride_train = config.max_seq_len_train if "max_seq_len_train" in config else None

        self.init_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "collate_fn": self.collate_fn
        }

        self.create_clusters = Build_Clusters(config)
        
        self.d1_items_array = np.array([])
        self.d2_items_array = np.array([])

        self.generate_cv_data(data)
        
        def create_dataset(i1, b, i2, d, ans, m1, m2, r_len, a_cnt):
            return TensorDataset(
                    torch.Tensor(i1).long(), torch.Tensor(b).long(),
                    torch.Tensor(i2).long(), torch.Tensor(d).long(),
                    torch.Tensor(ans).long(), torch.Tensor(m1).bool(), torch.Tensor(m2).bool(),
                    torch.Tensor(r_len).long(), torch.Tensor(a_cnt).long()
                )

        self.train_data = create_dataset(
            self.train_data_i1, self.train_data_b, self.train_data_i2, self.train_data_d,
            self.train_target_answers, self.train_target_masks, self.train_target_masks_i2,
            self.train_raw_lens, self.train_agg_counts
        )
        
        self.val_data = create_dataset(
            self.val_data_i1, self.val_data_b, self.val_data_i2, self.val_data_d,
            self.val_target_answers, self.val_target_masks, self.val_target_masks_i2,
            self.val_raw_lens, self.val_agg_counts
        )

        self.test_data = create_dataset(
            self.test_data_i1, self.test_data_b, self.test_data_i2, self.test_data_d,
            self.test_target_answers, self.test_target_masks, self.test_target_masks_i2,
            self.test_raw_lens, self.test_agg_counts
        )
        
        # create batched data
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size)
        self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size)
        # self.val_loader = DataLoader(self.val_data, batch_size=self.val_data_b.shape[0])
        # self.test_loader = DataLoader(self.test_data, batch_size=self.test_data_b.shape[0])

    def _get_seq_lengths(self, i1_data, i2_data):
        """Helper to count non-padding items in sequences."""
        valid_mask = (i1_data != 0) | (i2_data != 0)
        return valid_mask.sum(axis=1)

    def _get_stats(self, i1_data, b_data, i2_data):
        """
        Helper to calculate statistics for a given data split.
        """
        
        valid_mask_i1 = (i1_data != 0)
        valid_mask_i2 = (i2_data != 0)
        
        # Filter for valid interactions (non-padding)
        valid_interactions = valid_mask_i1 | valid_mask_i2
        
        # Views are where behavior is 0 AND it's a valid interaction
        views_mask = (b_data == 0) & valid_interactions
        total_views = views_mask.sum()
        
        # Unique Items
        unique_items_i1 = np.unique(i1_data[valid_mask_i1])
        unique_items_i2 = np.unique(i2_data[valid_mask_i2])
        unique_items_total = len(np.unique(np.concatenate([unique_items_i1, unique_items_i2])))
        
        # Average Sequence Length (non-padding count)
        # Per row count of valid interactions
        seq_lengths = valid_interactions.sum(axis=1)
        avg_seq_len = seq_lengths.mean() if len(seq_lengths) > 0 else 0.0
        
        return total_views, unique_items_total, avg_seq_len

    def generate_cv_data(self, data):
        """
        Processes data for K-Fold Cross Validation.
        Merges existing Train/Test splits, then applies KFold logic.
        """
        print(f"\n--- GENERATING {self.num_folds}-FOLD CV DATA (FOLD {self.fold_cv}) ---")

        # Merge existing Train/Test data from pickle to get a Full Dataset
        i1_records = data["traindata"]["i1_data"] + data["testdata"]["i1_data"]
        b_records = data["traindata"]["b_data"] + data["testdata"]["b_data"]
        i2_records = data["traindata"]["i2_data"] + data["testdata"]["i2_data"]
        d_records = data["traindata"]["d_data"] + data["testdata"]["d_data"]
        
        # Extract Unique Items per Domain
        d1_set = set()
        for seq in i1_records:
            for item in seq:
                if item != 0: d1_set.add(item)
        
        d2_set = set()
        for seq in i2_records:
            for item in seq:
                if item != 0: d2_set.add(item)
                
        self.d1_items_array = np.array(list(d1_set))
        self.d2_items_array = np.array(list(d2_set))
        
        print(f"Mixed Domain Detection: Found {len(self.d1_items_array)} D1 items and {len(self.d2_items_array)} D2 items.")
        
        total_users = len(i1_records)
        print(f"Total Users available: {total_users}")

        if self.mode == "train":
            # K-Fold Splitting AT THE USER LEVEL (before any windowing)
            user_indices = np.arange(total_users)
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.config.seed)

            target_fold_idx = self.fold_cv - 1  # Convert 1-based config to 0-based index

            train_user_idx, test_user_idx = None, None
            for fold, (train_i, test_i) in enumerate(kf.split(user_indices)):
                if fold == target_fold_idx:
                    train_user_idx, test_user_idx = train_i, test_i
                    break

            if train_user_idx is None:
                raise ValueError(f"Fold {self.fold_cv} is out of range for {self.num_folds} folds.")

            # Inner Split: Train Users (80%) -> Train Users (64%) / Val Users (16%)
            train_user_idx, val_user_idx = train_test_split(
                train_user_idx,
                test_size=self.config.validation_split,
                random_state=self.config.seed,
                shuffle=self.config.shuffle
            )

            print(f"Fold {self.fold_cv}: Train Users {len(train_user_idx)} | "
                  f"Val Users {len(val_user_idx)} | Test Users {len(test_user_idx)}")

            def _subset(idx):
                return (
                    [i1_records[i] for i in idx],
                    [b_records[i] for i in idx],
                    [i2_records[i] for i in idx],
                    [d_records[i] for i in idx],
                )

            i1_tr, b_tr, i2_tr, d_tr = _subset(train_user_idx)
            i1_va, b_va, i2_va, d_va = _subset(val_user_idx)
            i1_te, b_te, i2_te, d_te = _subset(test_user_idx)

            # Window EACH user-subset INDEPENDENTLY
            self.train_data_i1, self.train_data_b, self.train_data_i2, self.train_data_d = self.SBAC_ExtDataset(
                i1_tr, b_tr, i2_tr, d_tr, self.max_seq_len, stride=self.stride
            )
            self.val_data_i1, self.val_data_b, self.val_data_i2, self.val_data_d = self.SBAC_ExtDataset(
                i1_va, b_va, i2_va, d_va, self.max_seq_len, stride=self.stride
            )
            self.test_data_i1, self.test_data_b, self.test_data_i2, self.test_data_d = self.SBAC_ExtDataset(
                i1_te, b_te, i2_te, d_te, self.max_seq_len, stride=self.stride
            )

            self.train_target_answers = np.copy(self.train_data_b)
            self.train_target_masks = (self.train_data_i1 != 0)
            self.train_target_masks_i2 = (self.train_data_i2 != 0)

            self.val_target_answers = np.copy(self.val_data_b)
            self.val_target_masks = (self.val_data_i1 != 0)
            self.val_target_masks_i2 = (self.val_data_i2 != 0)

            self.test_target_answers = np.copy(self.test_data_b)
            self.test_target_masks = (self.test_data_i1 != 0)
            self.test_target_masks_i2 = (self.test_data_i2 != 0)
            
            # PRE-AGGREGATION STATISTICS
            print("\n" + "="*20 + " PRE-AGGREGATION STATS " + "="*20)
            train_views_pre, train_uniq_pre, train_len_pre = self._get_stats(self.train_data_i1, self.train_data_b, self.train_data_i2)
            val_views_pre, val_uniq_pre, val_len_pre = self._get_stats(self.val_data_i1, self.val_data_b, self.val_data_i2)
            test_views_pre, test_uniq_pre, test_len_pre = self._get_stats(self.test_data_i1, self.test_data_b, self.test_data_i2)
            
            print(f"TRAIN | Views: {train_views_pre} | Unique Items: {train_uniq_pre} | Avg Seq Len (No Pad): {train_len_pre:.2f}")
            print(f"VALID | Views: {val_views_pre} | Unique Items: {val_uniq_pre} | Avg Seq Len (No Pad): {val_len_pre:.2f}")
            print(f"TEST  | Views: {test_views_pre} | Unique Items: {test_uniq_pre} | Avg Seq Len (No Pad): {test_len_pre:.2f}")
            
            print("="*65 + "\n")

            # CALCULATE RAW LENGTHS (Before Aggregation)
            print("Calculating Raw Lengths...")
            self.train_raw_lens = self._get_seq_lengths(self.train_data_i1, self.train_data_i2)
            self.val_raw_lens = self._get_seq_lengths(self.val_data_i1, self.val_data_i2)
            self.test_raw_lens = self._get_seq_lengths(self.test_data_i1, self.test_data_i2)
            
            # AGGREGATION LOGIC (On the specific fold's training data)
            self.cluster_labels_d1 = None
            self.cluster_labels_d2 = None
            self.max_item_id_d1 = 0
            self.max_item_id_d2 = 0

            # DOMAIN 1 CLUSTERING
            if self.aggregation_mode in ['d1', 'both']:
                print(f"Building Clusters for Domain 1 (I1) [Fold {self.fold_cv}]...")
                fold_suffix = f"_d1_fold{self.fold_cv}"
                
                self.max_item_id_d1 = self.create_clusters.build_clusters(
                    self.train_data_i1, self.train_data_b, 
                    domain_suffix=fold_suffix,
                    n_clusters=self.config.n_clusters_d1,
                    window_size=self.config.window_size_d1,
                    w2v_epoch=self.config.w2v_epoch_d1
                )
                self.cluster_labels_d1 = self.create_clusters.load_cluster_labels(
                    domain_suffix=fold_suffix,
                    n_clusters=self.config.n_clusters_d1,
                    window_size=self.config.window_size_d1,
                    w2v_epoch=self.config.w2v_epoch_d1
                )

            # DOMAIN 2 CLUSTERING
            if self.aggregation_mode in ['d2', 'both']:
                print(f"Building Clusters for Domain 2 (I2) [Fold {self.fold_cv}]...")
                fold_suffix = f"_d2_fold{self.fold_cv}"
                
                self.max_item_id_d2 = self.create_clusters.build_clusters(
                    self.train_data_i2, self.train_data_b, 
                    domain_suffix=fold_suffix,
                    n_clusters=self.config.n_clusters_d2,
                    window_size=self.config.window_size_d2,
                    w2v_epoch=self.config.w2v_epoch_d2
                )
                self.cluster_labels_d2 = self.create_clusters.load_cluster_labels(
                    domain_suffix=fold_suffix,
                    n_clusters=self.config.n_clusters_d2,
                    window_size=self.config.window_size_d2,
                    w2v_epoch=self.config.w2v_epoch_d2
                )

            # AGGREGATE splits
            print(f"Aggregating data (Mode: {self.aggregation_mode})...")
            
            self.train_data_i1, self.train_data_b, self.train_data_i2, self.train_data_d = self.SBAC_ExtDataset_Agg(
                self.train_data_i1, self.train_data_b, self.train_data_i2, self.train_data_d,
                self.max_seq_len_train, stride=self.stride_train
            )
            self.train_target_answers = np.copy(self.train_data_b)
            self.train_target_masks = (self.train_data_i1 != 0)
            self.train_target_masks_i2 = (self.train_data_i2 != 0)

            print("Aggregating validation data...")
            self.val_data_i1, self.val_data_b, self.val_data_i2, self.val_data_d = self.SBAC_ExtDataset_Agg(
                self.val_data_i1, self.val_data_b, self.val_data_i2, self.val_data_d,
                self.max_seq_len_train, stride=self.stride_train
            )
            self.val_target_answers = np.copy(self.val_data_b)
            self.val_target_masks = (self.val_data_i1 != 0)
            self.val_target_masks_i2 = (self.val_data_i2 != 0)

            print("Aggregating test data...")
            self.test_data_i1, self.test_data_b, self.test_data_i2, self.test_data_d = self.SBAC_ExtDataset_Agg(
                self.test_data_i1, self.test_data_b, self.test_data_i2, self.test_data_d,
                self.max_seq_len_train, stride=self.stride_train
            )
            self.test_target_answers = np.copy(self.test_data_b)
            self.test_target_masks = (self.test_data_i1 != 0)
            self.test_target_masks_i2 = (self.test_data_i2 != 0)

            # CALCULATE AGGREGATION STATS
            # Agg Count = Raw Length - Aggregated Length
            train_agg_lens = self._get_seq_lengths(self.train_data_i1, self.train_data_i2)
            self.train_agg_counts = self.train_raw_lens - train_agg_lens
            
            val_agg_lens = self._get_seq_lengths(self.val_data_i1, self.val_data_i2)
            self.val_agg_counts = self.val_raw_lens - val_agg_lens
            
            test_agg_lens = self._get_seq_lengths(self.test_data_i1, self.test_data_i2)
            self.test_agg_counts = self.test_raw_lens - test_agg_lens
            
            print(f"Stats: Avg Aggregations per Seq -> Train: {self.train_agg_counts.mean():.2f}, Val: {self.val_agg_counts.mean():.2f}, Test: {self.test_agg_counts.mean():.2f}")

            # POST-AGGREGATION STATISTICS
            print("\n" + "="*20 + " POST-AGGREGATION STATS " + "="*20)
            train_views_post, train_uniq_post, train_len_post = self._get_stats(self.train_data_i1, self.train_data_b, self.train_data_i2)
            val_views_post, val_uniq_post, val_len_post = self._get_stats(self.val_data_i1, self.val_data_b, self.val_data_i2)
            test_views_post, test_uniq_post, test_len_post = self._get_stats(self.test_data_i1, self.test_data_b, self.test_data_i2)
            
            print(f"TRAIN | Views: {train_views_post} (Removed: {train_views_pre - train_views_post}) | Unique Items: {train_uniq_post} | Avg Seq Len (No Pad): {train_len_post:.2f}")
            print(f"VALID | Views: {val_views_post} (Removed: {val_views_pre - val_views_post}) | Unique Items: {val_uniq_post} | Avg Seq Len (No Pad): {val_len_post:.2f}")
            print(f"TEST  | Views: {test_views_post} (Removed: {test_views_pre - test_views_post}) | Unique Items: {test_uniq_post} | Avg Seq Len (No Pad): {test_len_post:.2f}")

            print("="*65 + "\n")

        elif self.mode == "test":
            pass
    
    def SBAC_ExtDataset(self, i1_records, b_records, i2_records, d_records, max_seq_len, stride):
        i1_data, b_data, i2_data, d_data = [], [], [], []
        for index in range(len(i1_records)):
            i1_list = list(i1_records[index])
            b_list = list(b_records[index])
            i2_list = list(i2_records[index])
            d_list = list(d_records[index])

            # Pad beginning
            for _ in range(2):
                i1_list.insert(0, 0)
                b_list.insert(0, 3) # padding
                i2_list.insert(0, 0)
                d_list.insert(0, 0)

            i1_patches = list(miter.windowed(i1_list, max_seq_len, fillvalue=0, step=stride-2))
            b_patches = list(miter.windowed(b_list, max_seq_len, fillvalue=3, step=stride-2))
            d_patches = list(miter.windowed(d_list, max_seq_len, fillvalue=0, step=stride-2))
            i2_patches = list(miter.windowed(i2_list, max_seq_len, fillvalue=0, step=stride-2))

            i1_data.extend(i1_patches)
            b_data.extend(b_patches)
            i2_data.extend(i2_patches)
            d_data.extend(d_patches)
        
        return np.array(i1_data), np.array(b_data), np.array(i2_data), np.array(d_data)

    def SBAC_ExtDataset_Agg(self, i1_records, b_records, i2_records, d_records, max_seq_len, stride):
        """
        Aggregates consecutive views in the same cluster.
        """
        i1_data, b_data, i2_data, d_data = [], [], [], []
        view_behavior_id = 0 
        
        agg_d1 = (self.cluster_labels_d1 is not None)
        agg_d2 = (self.cluster_labels_d2 is not None)

        for index in range(len(i1_records)):
            i1_list = list(i1_records[index])
            b_list = list(b_records[index])
            i2_list = list(i2_records[index])
            d_list = list(d_records[index])
            
            i1_agg, b_agg, i2_agg, d_agg = [], [], [], []
            
            t = 0
            while t < len(i1_list):
                i1_t = i1_list[t]
                i2_t = i2_list[t]
                b_t = b_list[t]
                d_t = d_list[t]
                
                # Default to no aggregation for this step
                should_aggregate = False
                
                # Check Domain 1 Aggregation
                if agg_d1 and i1_t > 0:
                    if i1_t > self.max_item_id_d1:
                        c_id = -1
                    else:
                        c_id = self.cluster_labels_d1[i1_t]
                    
                    if c_id != -1 and b_t == view_behavior_id and t < len(i1_list) - 1:
                        i1_next = i1_list[t+1]
                        b_next = b_list[t+1]
                        
                        if b_next == view_behavior_id and i1_next > 0:
                            if i1_next > self.max_item_id_d1:
                                next_c_id = -1
                            else:
                                next_c_id = self.cluster_labels_d1[i1_next]
                            
                            if next_c_id == c_id:
                                should_aggregate = True
                                # print("aggregation trigerred")
                
                # Check Domain 2 Aggregation
                if not should_aggregate and agg_d2 and i2_t > 0:
                    if i2_t > self.max_item_id_d2:
                        c_id = -1
                    else:
                        c_id = self.cluster_labels_d2[i2_t]
                    
                    if c_id != -1 and b_t == view_behavior_id and t < len(i1_list) - 1:
                        i2_next = i2_list[t+1]
                        b_next = b_list[t+1]
                        
                        if b_next == view_behavior_id and i2_next > 0:
                            if i2_next > self.max_item_id_d2:
                                next_c_id = -1
                            else:
                                next_c_id = self.cluster_labels_d2[i2_next]
                            
                            if next_c_id == c_id:
                                should_aggregate = True

                # Execute Aggregation or Append
                if should_aggregate:
                    # Find the end of the run
                    run_end_t = t + 1
                    while run_end_t < len(i1_list):
                        curr_i1 = i1_list[run_end_t]
                        curr_i2 = i2_list[run_end_t]
                        curr_b = b_list[run_end_t]
                        
                        if curr_b != view_behavior_id:
                            break
                        
                        if i1_t > 0: 
                            if curr_i1 == 0: break 
                            if curr_i1 > self.max_item_id_d1:
                                curr_cid = -1
                            else:
                                curr_cid = self.cluster_labels_d1[curr_i1]
                            if curr_cid != c_id: break
                        elif i2_t > 0:
                            if curr_i2 == 0: break 
                            if curr_i2 > self.max_item_id_d2:
                                curr_cid = -1
                            else:
                                curr_cid = self.cluster_labels_d2[curr_i2]
                            if curr_cid != c_id: break

                        run_end_t += 1
                    
                    last_idx = run_end_t - 1
                    i1_agg.append(i1_list[last_idx])
                    i2_agg.append(i2_list[last_idx])
                    b_agg.append(b_list[last_idx])
                    d_agg.append(d_list[last_idx])
                    
                    t = run_end_t 
                else:
                    i1_agg.append(i1_t)
                    i2_agg.append(i2_t)
                    b_agg.append(b_t)
                    d_agg.append(d_t)
                    t += 1

            i1_patches = list(miter.windowed(i1_agg, max_seq_len, fillvalue=0, step=stride-2))
            b_patches = list(miter.windowed(b_agg, max_seq_len, fillvalue=3, step=stride-2))
            d_patches = list(miter.windowed(d_agg, max_seq_len, fillvalue=0, step=stride-2))
            i2_patches = list(miter.windowed(i2_agg, max_seq_len, fillvalue=0, step=stride-2))

            i1_data.extend(i1_patches)
            b_data.extend(b_patches)
            i2_data.extend(i2_patches)
            d_data.extend(d_patches)

        return np.array(i1_data), np.array(b_data), np.array(i2_data), np.array(d_data)
