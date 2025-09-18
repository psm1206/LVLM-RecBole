# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
import os
import numpy as np
import json
import pickle
from sklearn.decomposition import PCA
# import torch.nn.functional as F

class Dong_SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(Dong_SASRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)
        
        ## LLM embedding load
        id_map = json.load(open(f'./dataset/{config["dataset"]}/id_map.json', "r"))["item2id"]
        loaded_feat = pickle.load(open(f'./dataset/{config["dataset"]}/{config["text_encoder"]}.pkl', "rb"))
        mapped_feat = np.zeros((self.n_items, loaded_feat.shape[1]), dtype=np.float32)
        for i, token in enumerate(dataset.field2id_token['item_id']):
            if token == '[PAD]': continue
            token_idx = int(id_map[token])-1
            mapped_feat[i] = loaded_feat[token_idx]
        
        
        # print("LAE_Item_Embedding init")
        if "Random" in config["desc"]:
            self.item_embedding = Random_Embedding(config, mapped_feat.T, self.n_items, self.hidden_size, scale=config['scale'], padding_idx=0, )
        else:
            self.item_embedding = LAE_Item_Embedding(config, mapped_feat.T, self.n_items, self.hidden_size, scale=config['scale'], padding_idx=0, )
            
        self.desc=config["desc"]
        # e = self.item_embedding.weight.clone().detach().cpu()
        # path = f'tmp/item_emb_{self.desc}'
        # os.makedirs(f'{path}', exist_ok=True)
        # with open(f'{path}/before.pkl', 'wb') as f:
        #     pickle.dump(e, f)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        # input_emb = item_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            # from IPython import embed; embed()
            # test_item_emb = self.item_embedding.norm_weight()
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            # logits /= self.temperature
            loss = self.loss_fct(logits, pos_items)
            return loss

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



class LAE_Item_Embedding(torch.nn.Module):
    def __init__(self, config, F, item_num, hidden_size, scale=1.0, padding_idx=0, ):
        super(LAE_Item_Embedding, self).__init__()
        print("LAE_Item_Embedding init")
        
        pca_desc = ""
        
        if config["PCA"]==True:
            F[:,1:] = F[:,1:] - F[:,1:].mean(axis=1, keepdims=True)
            pca_desc = "_PCA"
            
        GF = F.T@F        
        
        self.trade_off = config["trade_off"]
        self.reg_weight = config["reg_weight"]
        self.target_norm = config["target_norm"]
        
        from scipy.linalg import eigh
        
        G = GF
        
        self.dataset_name = config["dataset"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size

        print("eighing")
        if os.path.exists(f'eigvals_{self.dataset_name}{pca_desc}.npy'):
            w = np.load(f'eigvals_{self.dataset_name}{pca_desc}.npy')
            U = np.load(f'eigvecs_{self.dataset_name}{pca_desc}.npy')
        else:
            w, U = eigh(G[1:,1:])
            
            np.save(f'eigvals_{self.dataset_name}{pca_desc}.npy', w)
            np.save(f'eigvecs_{self.dataset_name}{pca_desc}.npy', U)
        
        
        U_k = U[:, -self.hidden_size:]
        w_k = w[-self.hidden_size:]
        
        
        if config["PCA"]==True:
            print(f"[PCA_init_weights], {F.shape}, {U_k.shape}, {w_k.shape}")
            sqrt_lambdas = np.sqrt(w_k)
        else:
            sqrt_lambdas = np.sqrt(w_k / (w_k + self.reg_weight))
            
        X = U_k * sqrt_lambdas[np.newaxis, :]        
        
        print(f"X.mean(): {X.mean():.6f}")
        print(f"X.std(): {X.std():.6f}")
        print(f"1/X.std(): {(1/X.std()):.6f}")
        print(f"X.norm: {(np.linalg.norm(X, axis=1).mean()):.6f}")
        print(f"X_scaled.norm: {(np.linalg.norm(X, axis=1).mean()/X.std()):.6f}")
        
        # from IPython import embed; embed()
        
        if config["norm"]:
            # X_scaled = (X - X.mean()) / (X.std() + 1e-6) * scale
            X_scaled = (X) / (X.std() + 1e-6) * scale
        else:
            # X_scaled = X * 100 / np.linalg.norm(X) * scale
            X_scaled = X * scale
                
        self.item_embedding = nn.Embedding(
            item_num, hidden_size, padding_idx=0
        )        
        X_full = torch.zeros((item_num, self.hidden_size), dtype=torch.float,
                            device=self.item_embedding.weight.device)
        X_full[1:] = torch.from_numpy(X_scaled).float()
        with torch.no_grad():
            self.item_embedding.weight.copy_(X_full)
            # self.item_embedding.requires_grad = False
        
        self.forward_norm = config["forward_norm"]
        
        
    ### @property
    def forward(self, item_ids):
        item_embedding = self.item_embedding(item_ids)
        # if self.forward_norm and self.target_norm is not None:
        # item_embedding = self.target_norm * item_embedding / (item_embedding.norm(dim=-1, keepdim=True) + 1e-12)
        return item_embedding
    
    @property
    def weight(self):
        # 0 ~ item_num (padding 포함)까지 한번에 임베딩 조회
        all_ids = torch.arange(self.item_embedding.num_embeddings, device=self.item_embedding.weight.device)
        e = self(all_ids)
        if self.target_norm is not None:
            e = self.target_norm * e / (e.norm(dim=-1, keepdim=True) + 1e-12)
        return e
    # # @property
    # def norm_weight(self):
    #     # 0 ~ item_num (padding 포함)까지 한번에 임베딩 조회
    #     all_ids = torch.arange(self.item_embedding.num_embeddings, device=self.item_embedding.weight.device)
    #     e = self(all_ids)
    #     if self.target_norm is not None:
    #         e = self.target_norm * e / (e.norm(dim=1, keepdim=True) + 1e-12)
    #     return e
    
class Random_Embedding(torch.nn.Module):
    def __init__(self, config, F, item_num, hidden_size, scale=1.0, padding_idx=0, ):
        super(Random_Embedding, self).__init__()
        print("Random_Embedding init")
        
                
        self.item_embedding = nn.Embedding(
            item_num, hidden_size, padding_idx=0
        )        
        with torch.no_grad():
            self.item_embedding.weight.normal_(mean=0.0, std=0.01)
            self.item_embedding.weight *= scale
        self.target_norm = config["target_norm"]
        
    ### @property
    def forward(self, item_ids):
        item_embedding = self.item_embedding(item_ids)
        return item_embedding
    
    @property
    def weight(self):
        # 0 ~ item_num (padding 포함)까지 한번에 임베딩 조회
        all_ids = torch.arange(self.item_embedding.num_embeddings, device=self.item_embedding.weight.device)
        e = self(all_ids)
        if self.target_norm is not None:
            e = self.target_norm * e / (e.norm(dim=1, keepdim=True) + 1e-12)
        return e