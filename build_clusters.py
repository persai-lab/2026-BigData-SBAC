import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
from easydict import EasyDict
import warnings
import os
import seaborn as sns
from gensim.models import Word2Vec

class Build_Clusters():
    def __init__(self, config):
        super(Build_Clusters, self).__init__()
        self.config = config
        self.data_name = self.config.data_name
        self.seed = self.config.seed
        
        np.random.seed(self.seed)
    
    def build_clusters(self, item_data, b_data, domain_suffix="_d1", n_clusters=10, window_size=5, w2v_epoch=5):
        print(f"\n[{domain_suffix.upper()}] Building Clusters...")
        print(f"[{domain_suffix.upper()}] Params: n_clusters={n_clusters}, window={window_size}, epochs={w2v_epoch}")
        
        # Get all sequences
        item_records = item_data
        b_records = b_data

        # Find max item ID to size things correctly
        max_item_id = 0
        for seq in item_records:
            if len(seq) > 0:
                curr_max = max(seq)
                if curr_max > max_item_id:
                    max_item_id = curr_max
        
        print(f"[{domain_suffix}] Max item ID found in data: {max_item_id}.")

        # Prepare Sentences for Word2Vec
        print(f"[{domain_suffix}] Preparing item sequences for Word2Vec...")
        view_behavior_id = 0
        sentences = []
        
        valid_item_ids = set()

        for i_list, b_list in zip(item_records, b_records):
            # Filter sequence to only keep 'view' items
            view_sequence = []
            for item, action in zip(i_list, b_list):
                if action == view_behavior_id and item > 0:
                    item_str = str(item)
                    view_sequence.append(item_str)
                    valid_item_ids.add(item)
            
            if len(view_sequence) > 1:
                sentences.append(view_sequence)
        
        print(f"[{domain_suffix}] Training Word2Vec on {len(sentences)} sequences...")
        print(f"[{domain_suffix}] Unique items found in view sequences: {len(valid_item_ids)}")

        if len(sentences) == 0:
            print(f"[{domain_suffix}] WARNING: No view sequences found. Skipping clustering.")
            return max_item_id

        # Train Word2Vec (Item2Vec)
        w2v_model = Word2Vec(
            sentences, 
            vector_size=64, 
            window=window_size, 
            min_count=1, 
            workers=1,
            seed=self.seed,
            epochs=w2v_epoch
        )
        
        print(f"[{domain_suffix}] Word2Vec training complete.")

        # Extract Embeddings for Clustering
        valid_embeddings = []
        valid_ids_list = [] 

        for item_id in sorted(list(valid_item_ids)):
            item_str = str(item_id)
            if item_str in w2v_model.wv:
                valid_embeddings.append(w2v_model.wv[item_str])
                valid_ids_list.append(item_id)
        
        valid_embeddings = np.array(valid_embeddings)
        print(f"[{domain_suffix}] Created embedding matrix for clustering. Shape: {valid_embeddings.shape}")

        # K-Means Clustering
        print(f"[{domain_suffix}] Starting K-Means Clustering with {n_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        cluster_labels_valid = kmeans.fit_predict(valid_embeddings)
        
        # Map back to full ID range
        final_cluster_labels = np.full(max_item_id + 1, -1, dtype=np.int32)
        
        for idx, item_id in enumerate(valid_ids_list):
            final_cluster_labels[item_id] = cluster_labels_valid[idx]
            
        output_path = f'data/{self.config.data_name}/cluster_labels{domain_suffix}_n{n_clusters}_window_{window_size}_w2v_epoch_{w2v_epoch}_view_to_view.npy'

        np.save(output_path, final_cluster_labels)
        print(f"[{domain_suffix}] Clustering complete. Saved to: {output_path}")
        
        return max_item_id

    def load_cluster_labels(self, domain_suffix="_d1", n_clusters=10, window_size=5, w2v_epoch=5):
        cluster_labels_path = f'data/{self.config.data_name}/cluster_labels{domain_suffix}_n{n_clusters}_window_{window_size}_w2v_epoch_{w2v_epoch}_view_to_view.npy'
        if not os.path.exists(cluster_labels_path):
            print(f"Error: Could not find cluster file at {cluster_labels_path}")
            return None
        print(f"Loading cluster labels from {cluster_labels_path}")
        cluster_labels = np.load(cluster_labels_path)
        return cluster_labels
