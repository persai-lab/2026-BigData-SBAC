import torch
from torch import nn
import math

class SBAC(nn.Module):
    def __init__(self, config):
        super(SBAC, self).__init__()
        self.config = config
        self.device = torch.device(f"cuda:{config.gpu_device}" if config.cuda and torch.cuda.is_available() else "cpu")
        
        # Dimensions
        self.num_items = config.num_items
        self.num_domains = config.num_domains
        self.hidden_size = config.hidden_size
        self.embedding_size_item = config.embedding_size_item
        self.embedding_size_b = config.embedding_size_b
        self.embedding_size_domain = config.embedding_size_domain
        self.dropout_rate = getattr(config, 'dropout_rate', 0.1)
        
        # EMBEDDINGS
        self.item_emb_shared = nn.Embedding(self.num_items + 1, self.embedding_size_item, padding_idx=0)
        self.item_emb_rank_d1 = nn.Embedding(self.num_items + 1, self.embedding_size_item, padding_idx=0)
        self.item_emb_rank_d2 = nn.Embedding(self.num_items + 1, self.embedding_size_item, padding_idx=0)
        self.item_emb_beh_d1 = nn.Embedding(self.num_items + 1, self.embedding_size_item, padding_idx=0)
        self.item_emb_beh_d2 = nn.Embedding(self.num_items + 1, self.embedding_size_item, padding_idx=0)
        
        self.b_embed_matrix_shared = nn.Embedding(4, self.embedding_size_b, padding_idx=3)
        self.b_embed_matrix_d1 = nn.Embedding(4, self.embedding_size_b, padding_idx=3)
        self.b_embed_matrix_d2 = nn.Embedding(4, self.embedding_size_b, padding_idx=3)
        self.domain_embed_matrix_shared = nn.Embedding(self.num_domains, self.embedding_size_domain)
        
        input_feature_dim = self.embedding_size_item + self.embedding_size_b + self.embedding_size_domain
        
        self.proj_shared = nn.Linear(input_feature_dim, self.hidden_size)
        self.proj_rank_d1 = nn.Linear(input_feature_dim, self.hidden_size)
        self.proj_rank_d2 = nn.Linear(input_feature_dim, self.hidden_size)
        self.proj_beh_d1 = nn.Linear(input_feature_dim, self.hidden_size)
        self.proj_beh_d2 = nn.Linear(input_feature_dim, self.hidden_size)
        
        self.position_embedding = nn.Embedding(config.max_seq_len, self.hidden_size)
        
        # EXPERTS
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=config.get('num_heads', 2),
            dim_feedforward=config.hidden_size * 4,
            dropout=self.dropout_rate,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        
        self.expert_shared = nn.TransformerEncoder(encoder_layer, num_layers=config.get('num_blocks', 2))
        self.expert_rank_d1 = nn.TransformerEncoder(encoder_layer, num_layers=config.get('num_blocks', 2))
        self.expert_rank_d2 = nn.TransformerEncoder(encoder_layer, num_layers=config.get('num_blocks', 2))
        self.expert_beh_d1 = nn.TransformerEncoder(encoder_layer, num_layers=config.get('num_blocks', 2))
        self.expert_beh_d2 = nn.TransformerEncoder(encoder_layer, num_layers=config.get('num_blocks', 2))
        
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_rate)

        # GATING
        self.gate_rank_d1 = nn.Sequential(
            nn.Linear(self.hidden_size * 5, 5),
            nn.Softmax(dim=-1)
        )
        self.gate_rank_d2 = nn.Sequential(
            nn.Linear(self.hidden_size * 5, 5),
            nn.Softmax(dim=-1)
        )
        self.gate_beh_d1 = nn.Sequential(
            nn.Linear(self.hidden_size * 5, 5),
            nn.Softmax(dim=-1)
        )
        self.gate_beh_d2 = nn.Sequential(
            nn.Linear(self.hidden_size * 5, 5),
            nn.Softmax(dim=-1)
        )

        # PREDICTION TOWERS
        self.tower_behavior_d1 = nn.Sequential(
            nn.Linear(self.hidden_size + self.embedding_size_item, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        self.W_0h_d1 = nn.Linear(self.hidden_size, 3) 

        self.tower_behavior_d2 = nn.Sequential(
            nn.Linear(self.hidden_size + self.embedding_size_item, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        self.W_0h_d2 = nn.Linear(self.hidden_size, 3)
        
        self.tower_item_d1 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        self.W_item_d1 = nn.Linear(self.hidden_size, self.num_items + 1)

        self.tower_item_d2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        self.W_item_d2 = nn.Linear(self.hidden_size, self.num_items + 1)
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _process_stream(self, item_emb_table, proj_layer, encoder, item_ids, beh_emb, domain_emb, pos_emb, attn_mask, padding_mask):
        items = item_emb_table(item_ids)
        features = torch.cat([items, beh_emb, domain_emb], dim=-1)
        seq = proj_layer(features) * math.sqrt(self.hidden_size)
        seq = seq + pos_emb
        seq = self.dropout(self.layer_norm(seq))
        return encoder(seq, mask=attn_mask, src_key_padding_mask=padding_mask)

    def forward(self, i1_data, b_data, i2_data, d_data):
        batch_size, seq_len = i1_data.size()
        
        item_ids = i1_data + i2_data
        beh_emb_shared = self.b_embed_matrix_shared(b_data)
        beh_emb_d1 = self.b_embed_matrix_d1(b_data)
        beh_emb_d2 = self.b_embed_matrix_d2(b_data)
        domain_emb_shared = self.domain_embed_matrix_shared(d_data)
        
        positions = torch.arange(0, seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        
        key_padding_mask = (item_ids == 0)
        if key_padding_mask.size(1) > 0: key_padding_mask[:, 0] = False
        attn_mask = self._generate_square_subsequent_mask(seq_len).to(self.device)

        h_shared = self._process_stream(self.item_emb_shared, self.proj_shared, self.expert_shared, item_ids, beh_emb_shared, domain_emb_shared, pos_emb, attn_mask, key_padding_mask)
        h_rank_d1 = self._process_stream(self.item_emb_rank_d1, self.proj_rank_d1, self.expert_rank_d1, item_ids, beh_emb_d1, domain_emb_shared, pos_emb, attn_mask, key_padding_mask)
        h_rank_d2 = self._process_stream(self.item_emb_rank_d2, self.proj_rank_d2, self.expert_rank_d2, item_ids, beh_emb_d2, domain_emb_shared, pos_emb, attn_mask, key_padding_mask)
        h_beh_d1 = self._process_stream(self.item_emb_beh_d1, self.proj_beh_d1, self.expert_beh_d1, item_ids, beh_emb_d1, domain_emb_shared, pos_emb, attn_mask, key_padding_mask)
        h_beh_d2 = self._process_stream(self.item_emb_beh_d2, self.proj_beh_d2, self.expert_beh_d2, item_ids, beh_emb_d2, domain_emb_shared, pos_emb, attn_mask, key_padding_mask)

        # GATING
        gate_input = torch.cat([h_shared, h_rank_d1, h_rank_d2, h_beh_d1, h_beh_d2], dim=-1)
        
        weights_rank_d1 = self.gate_rank_d1(gate_input).unsqueeze(-1)
        stack_rank_d1 = torch.stack([h_shared, h_rank_d1, h_rank_d2.detach(), h_beh_d1.detach(), h_beh_d2.detach()], dim=2)
        out_rank_d1 = (stack_rank_d1 * weights_rank_d1).sum(dim=2)

        weights_rank_d2 = self.gate_rank_d2(gate_input).unsqueeze(-1)
        stack_rank_d2 = torch.stack([h_shared, h_rank_d2, h_rank_d1.detach(), h_beh_d1.detach(), h_beh_d2.detach()], dim=2)
        out_rank_d2 = (stack_rank_d2 * weights_rank_d2).sum(dim=2)
        
        weights_beh_d1 = self.gate_beh_d1(gate_input).unsqueeze(-1)
        stack_beh_d1 = torch.stack([h_shared, h_beh_d1, h_beh_d2.detach(), h_rank_d1.detach(), h_rank_d2.detach()], dim=2)
        out_beh_d1 = (stack_beh_d1 * weights_beh_d1).sum(dim=2)

        weights_beh_d2 = self.gate_beh_d2(gate_input).unsqueeze(-1)
        stack_beh_d2 = torch.stack([h_shared, h_beh_d2, h_beh_d1.detach(), h_rank_d1.detach(), h_rank_d2.detach()], dim=2)
        out_beh_d2 = (stack_beh_d2 * weights_beh_d2).sum(dim=2)

        # PREDICTION
        relevant_hidden_beh_d1 = out_beh_d1[:, :-1, :]
        relevant_hidden_beh_d2 = out_beh_d2[:, :-1, :]
        relevant_hidden_rank_d1 = out_rank_d1[:, :-1, :]
        relevant_hidden_rank_d2 = out_rank_d2[:, :-1, :]
        
        next_item_ids = item_ids[:, 1:]
        relevant_next_items = self.item_emb_shared(next_item_ids)
        
        beh_input_d1 = torch.cat([relevant_hidden_beh_d1, relevant_next_items], dim=-1)
        beh_latent_d1 = self.tower_behavior_d1(beh_input_d1)
        batch_pred_behavior_d1 = self.W_0h_d1(beh_latent_d1)

        beh_input_d2 = torch.cat([relevant_hidden_beh_d2, relevant_next_items], dim=-1)
        beh_latent_d2 = self.tower_behavior_d2(beh_input_d2)
        batch_pred_behavior_d2 = self.W_0h_d2(beh_latent_d2)
        
        item_latent_d1 = self.tower_item_d1(relevant_hidden_rank_d1)
        item_latent_d2 = self.tower_item_d2(relevant_hidden_rank_d2)
        
        return batch_pred_behavior_d1, batch_pred_behavior_d2, item_latent_d1, item_latent_d2
