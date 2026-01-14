# -*- coding: utf-8 -*-
# @Time    : 2025/10/15 21:43 KST
# @Author  : Seongmin Park
# @Email   : psm1206@skku.edu

"""
Zero-shot Recommendation with Extracted LVLM Embeddings

"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import json
import pickle
from sklearn.decomposition import PCA
import time
import os

from recbole.model.abstract_recommender import SequentialRecommender


class LVLM_Emb_ZeroShot_LateFusion(SequentialRecommender):
    r"""
    Zero-shot Recommendation with Extracted LVLM Embeddings
    [Note] LVLM Embedding won't be trained and directly used as item embedding.

    """

    def __init__(self, config, dataset):
        super(LVLM_Emb_ZeroShot_LateFusion, self).__init__(config, dataset)

        # load model info
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.agg_method = config["agg_method"]  # exp_decay or mean
        self.late_fusion_weight = config["late_fusion_weight"]

        ## LVLM embedding load
        dataset_path = f'./dataset/{config["dataset"]}'
        id_map = json.load(open(f'{dataset_path}/id_map.json', "r"))["item2id"]
        loaded_feat1 = pickle.load(open(f'{dataset_path}/{config["text_encoder"]}.pkl', "rb"))
        loaded_feat2 = pickle.load(open(f'{dataset_path}/{config["image_encoder"]}.pkl', "rb"))
        llm_emb_dim = loaded_feat1.shape[1]

        embedding_size = llm_emb_dim

        mapped_feat1 = np.zeros((self.n_items, embedding_size), dtype=np.float32)
        mapped_feat2 = np.zeros((self.n_items, embedding_size), dtype=np.float32)
        for i, token in enumerate(dataset.field2id_token['item_id']):
            if token == '[PAD]': continue
            token_idx = int(id_map[token])-1
            mapped_feat1[i] = loaded_feat1[token_idx]
            mapped_feat2[i] = loaded_feat2[token_idx]

        reduced_llm_item_emb1 = mapped_feat1[1:,:]
        reduced_llm_item_emb2 = mapped_feat2[1:,:]
        reduced_llm_item_emb1 = np.concatenate((np.zeros((1, embedding_size)), reduced_llm_item_emb1), axis=0)  # zero padding for [PAD] token
        reduced_llm_item_emb2 = np.concatenate((np.zeros((1, embedding_size)), reduced_llm_item_emb2), axis=0)  # zero padding for [PAD] token

        # late fusion with weighted sum 
        reduced_llm_item_emb = (reduced_llm_item_emb1 * self.late_fusion_weight + reduced_llm_item_emb2 * (1 - self.late_fusion_weight))

        # build item embedding as a frozen nn.Embedding (padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items, embedding_size, padding_idx=0)
        self.item_embedding.weight = nn.Parameter(
            torch.tensor(reduced_llm_item_emb, dtype=torch.float32), requires_grad=False
        )

    def forward(self, item_seq, item_seq_len):
        # item_seq: [B, L] - right padded (actual items in front, zeros at the end)
        item_emb = self.item_embedding(item_seq)  # [B, L, H]

        # masking for position weighting
        mask = (item_seq != 0).unsqueeze(-1).float()  # [B, L, 1]

        B, L = item_seq.size()
        last_idx = (item_seq_len - 1).unsqueeze(1)  # [B, 1]
        if self.agg_method == 'exp_decay':
            # per-sample position weighting based on sequence length
            # For a sequence with length T at max length L:
            # weights_row = [exp(-(T-1-0)), exp(-(T-1-1)), ..., exp(-(T-1-(T-1))), 0, ..., 0]
            # Example (L=6, T=4): [exp(-3), exp(-2), exp(-1), exp(-0), 0, 0]
            position_ids = torch.arange(L, device=item_seq.device).unsqueeze(0).expand(B, L)  # [B, L]
            exponent = last_idx - position_ids  # [B, L]
            exponent = torch.clamp(exponent, min=0)  # negative -> 0 (will be masked anyway)
            base_weights = torch.exp(-exponent)  # [B, L]
            weights = base_weights.unsqueeze(-1) * mask  # [B, L, 1]
        elif self.agg_method == 'last_item': # last item only
            position_ids = torch.arange(L, device=item_seq.device).unsqueeze(0)  # [1, L]
            one_hot = (position_ids == last_idx).unsqueeze(-1).float()      # [B, L, 1]
            weights = one_hot
        elif self.agg_method == 'mean': # mean of all items
            weights = mask
        else:
            raise ValueError(f"Invalid aggregation method: {self.agg_method}")
        
        weighted_sum_emb = (item_emb * weights).sum(dim=1)  # [B, H]
        denom = weights.sum(dim=1)  # [B, 1]
        output = weighted_sum_emb / denom  # [B, H]
        return output

    def calculate_loss(self, interaction):
        return torch.tensor(0.0)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
