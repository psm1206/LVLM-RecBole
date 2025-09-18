# -*- coding: utf-8 -*-
# @Time   : 2020/8/17 19:38
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

# UPDATE:
# @Time   : 2020/8/19, 2020/10/2
# @Author : Yupeng Hou, Yujie Lu
# @Email  : houyupeng@ruc.edu.cn, yujielu1998@gmail.com

r"""
GRU4Rec
################################################

Reference:
    Yong Kiam Tan et al. "Improved Recurrent Neural Networks for Session-based Recommendations." in DLRS 2016.

"""

import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_

import numpy as np
import json
import pickle
from sklearn.decomposition import PCA
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class LLM2X_Scaled_GRU4Rec(SequentialRecommender):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, config, dataset):
        super(LLM2X_Scaled_GRU4Rec, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]

        self.scale = config["scale"]        
        self.CE_loss_temperature = config["CE_loss_temperature"]
       
        # define layers and loss
        
        ## LLM embedding load
        id_map = json.load(open(f'./dataset/{config["dataset"]}/id_map.json', "r"))["item2id"]
        loaded_feat = pickle.load(open(f'./dataset/{config["dataset"]}/{config["text_encoder"]}.pkl', "rb"))
        mapped_feat = np.zeros((self.n_items, loaded_feat.shape[1]), dtype=np.float32)
        for i, token in enumerate(dataset.field2id_token['item_id']):
            if token == '[PAD]': continue
            token_idx = int(id_map[token])-1
            mapped_feat[i] = loaded_feat[token_idx]
        
        pca = PCA(n_components=self.embedding_size)
        reduced_llm_item_emb = pca.fit_transform(mapped_feat[1:,:])
        reduced_llm_item_emb = np.concatenate((np.zeros((1, self.embedding_size)), reduced_llm_item_emb), axis=0)  # zero padding for [PAD] token
        
        if config["UnitNorm"]:
            norms = np.linalg.norm(reduced_llm_item_emb, axis=1)
            reduced_llm_item_emb = reduced_llm_item_emb / (norms[:, None] + 1e-12)
        reduced_llm_item_emb *=  self.scale
        
        # item Emb initialization
        self.item_embedding = nn.Embedding.from_pretrained(torch.Tensor(reduced_llm_item_emb))
        self.item_embedding.weight.requires_grad = True  # allow fine-tuning of LLM embeddings
        # LLM Emb initialization
        self.llm_item_embedding = nn.Embedding.from_pretrained(torch.Tensor(mapped_feat))
        self.llm_item_embedding.weight.requires_grad = False  # freeze LLM embeddings, only
        
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        
        self.laser = config["laser"]
        if self.laser:
            self.kd_loss = nn.KLDivLoss(reduction='batchmean')
            self.alpha = config["alpha"]
            self.temperature = config["temperature"]
        
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding)) and module is not self.item_embedding and module is not self.llm_item_embedding:
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        return seq_output
    
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
            logits /= self.CE_loss_temperature
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
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores
