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
import torch.nn.functional as F

class Alphafuse_SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(Alphafuse_SASRec, self).__init__(config, dataset)

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
        
        ## LLM embedding load
        id_map = json.load(open(f'./dataset/{config["dataset"]}/id_map.json', "r"))["item2id"]
        loaded_feat = pickle.load(open(f'./dataset/{config["dataset"]}/{config["text_encoder"]}.pkl', "rb"))
        mapped_feat = np.zeros((self.n_items, loaded_feat.shape[1]), dtype=np.float32)
        from tqdm import tqdm
        for i, token in tqdm(enumerate(dataset.field2id_token['item_id'])):
            if token == '[PAD]': continue
            token_idx = int(id_map[token])-1
            mapped_feat[i] = loaded_feat[token_idx]


        # self.item_embedding = Item_Embedding(mapped_feat, self.n_items, self.hidden_size, init_type="uniform", ID_dim=int(self.hidden_size/2), scale=40, padding_idx=0)

        self.llm_item_embedding = nn.Embedding.from_pretrained(torch.Tensor(mapped_feat))
        self.llm_item_embedding.weight.requires_grad = False  # freeze LLM embeddings, only
        
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
        
        
        self.laser = config["laser"]
        if self.laser:
            self.kd_loss = nn.KLDivLoss(reduction='batchmean')
            self.alpha = config["alpha"]
            self.temperature = config["temperature"]

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
        self.item_embedding = Item_Embedding(mapped_feat, self.n_items, self.hidden_size, init_type="uniform", ID_dim=int(self.hidden_size/2), scale=40, padding_idx=0)


    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)) and module is not self.llm_item_embedding:
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
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]
    
    def reg_loss(self, target, user_emb):
        llm_feat = self.llm_item_embedding(target)
        teacher_embs = self.llm_item_embedding.weight
        student_embs = self.item_embedding.weight

        sim_teacher = torch.matmul(llm_feat, teacher_embs.transpose(0, 1)) / self.temperature # [B num_items]
        sim_student = torch.matmul(user_emb, student_embs.transpose(0, 1))

        log_probs = F.log_softmax(sim_student, dim=-1)
        target_probs = F.softmax(sim_teacher, dim=-1).detach() # teacher: soft target
        align_loss = self.kd_loss(log_probs, target_probs)
        return align_loss

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
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            if self.laser:
                loss += self.alpha * self.reg_loss(pos_items, seq_output)
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

class Item_Embedding(torch.nn.Module):
    def __init__(self, language_embs, item_num, hidden_size, init_type="uniform", ID_dim=64, scale=40, padding_idx=0):
        super(Item_Embedding, self).__init__()
        print("Item_Embedding init")
        
        self.nullity = ID_dim
        print(f"self.nullity: {self.nullity}, hidden_size: {hidden_size}, init_type: {init_type}, scale: {scale}")
        ### init ID embeddings ###
        ### init ID embeddings ###
        ### init ID embeddings ###
        self.ID_embeddings = nn.Embedding(
            num_embeddings=item_num+1, 
            # embedding_dim=hidden_size-ID_dim,
            embedding_dim=ID_dim,
            # padding_idx=0
        )
        if init_type == "uniform":
            nn.init.uniform_(self.ID_embeddings.weight, a=0.0, b=1.0)
        elif init_type == "normal":
            nn.init.normal_(self.ID_embeddings.weight, 0, 1)
        elif init_type == "zeros":
            nn.init.zeros_(self.ID_embeddings.weight)
        elif init_type == "ortho":
            nn.init.orthogonal_(self.ID_embeddings.weight, gain=1.0)
        elif init_type == "xavier":
            nn.init.xavier_uniform_(self.ID_embeddings.weight, gain=1.0)
        elif init_type == "sparse":
            nn.init.sparse_(self.ID_embeddings.weight, 0.01, std=1)
        else:
            raise NotImplementedError("This kind of init for ID embeddings is not implemented yet.")
        
        ### init LLM embeddings ###
        ### init LLM embeddings ###
        ### init LLM embeddings ###
        language_embs = language_embs[1:,:] * scale
        self.language_mean = np.mean(language_embs, axis=0)
        
        cov = np.cov(language_embs - self.language_mean, rowvar=False)
        U, S, _ = np.linalg.svd(cov, full_matrices=False)
        
        Projection_matrix = U[...,:hidden_size]
        Diagnals = np.sqrt(1/S)[:hidden_size]
        # Diagnals = 0.1*np.sqrt(1/S)[:hidden_size]
        
        Projection_matrix = Projection_matrix.dot(np.diag(Diagnals)) # V_{\lamda} into V_1
        clipped_language_embs = (language_embs-self.language_mean).dot(Projection_matrix)
        
        padding_emb = np.random.rand(clipped_language_embs.shape[1])  # padding ID embedding, padding_idx=0
        clipped_language_embs = np.vstack([padding_emb, clipped_language_embs]) # (self.item_num, 128)
        self.language_embeddings = torch.nn.Embedding.from_pretrained(
            torch.tensor(clipped_language_embs,dtype=torch.float32),
            freeze=True,
            padding_idx=0
        )
        
        
        
    ### @property
    def __call__(self, item_ids):
        language_embs = self.language_embeddings(item_ids)
        ID_embs = self.ID_embeddings(item_ids)
        fuse_embs = language_embs.clone()
        fuse_embs[...,-self.nullity:] = language_embs[...,-self.nullity:] + ID_embs
        return fuse_embs
    
    @property
    def weight(self):
        # 0 ~ item_num (padding 포함)까지 한번에 임베딩 조회
        all_ids = torch.arange(self.language_embeddings.num_embeddings, device=self.language_embeddings.weight.device)
        return self(all_ids)  # __call__(all_ids)를 통해 fuse된 임베딩 행렬 반환