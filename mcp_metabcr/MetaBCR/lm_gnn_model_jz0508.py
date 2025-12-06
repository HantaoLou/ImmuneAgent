import pandas as pd
import os
import numpy as np
import math
from torch.nn import Tanh, BatchNorm1d
from typing import Optional
# from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch
from transformers import BertModel
import torchvision





class RelativeMultiHeadAttention(torch.nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout
    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, feat, seq): Tensor containing query vector
        - **key** (batch, feat, seq): Tensor containing key vector
        - **value** (batch, feat, seq): Tensor containing value vector
        - **pos_embedding** (batch, feat, seq): Positional embedding tensor
        - **mask** (batch, 1, seq2) or (batch, seq1, seq2): Tensor containing indices to be masked
    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """

    def __init__(
            self,
            d_model: int = 1024,
            num_heads: int = 16,
            dropout_p: float = 0.,
            ndims: int = 1,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        # proj_layer=getattr(torch.nn,'Linear')
        # self.query_proj = proj_layer(d_model, d_model)
        # self.key_proj = proj_layer(d_model, d_model)
        # self.value_proj = proj_layer(d_model, d_model)
        # self.pos_proj = proj_layer(d_model, d_model, bias=False)
        # self.out_proj = proj_layer(d_model, d_model)
        proj_layer = getattr(torch.nn, 'Conv%dd' % ndims)
        self.query_proj = proj_layer(d_model, d_model, 1)
        self.key_proj = proj_layer(d_model, d_model, 1)
        self.value_proj = proj_layer(d_model, d_model, 1)
        self.pos_proj = proj_layer(d_model, d_model, 1, bias=False)
        self.out_proj = proj_layer(d_model, d_model, 1)

        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.u_bias = torch.nn.Parameter(torch.Tensor(self.num_heads, self.d_head, 1))
        self.v_bias = torch.nn.Parameter(torch.Tensor(self.num_heads, self.d_head, 1))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)
        self.Softmax_layer = torch.nn.Softmax(dim=-1)
        self.make_attn_einsum_eq = 'b h k i, b h k j -> b h i j'
        self.cal_attn_einsum_eq = 'b h i j, b h k j -> b h k i'

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            pos_embedding: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        print(query.shape)
        # query = self.query_proj(query).view(batch_size, self.num_heads, self.d_head, -1)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        # key = self.key_proj(key).view(batch_size, self.num_heads, self.d_head, -1)  # .permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        # value = self.value_proj(value).view(batch_size, self.num_heads, self.d_head, -1)  # .permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)
        # pos_embedding = self.pos_proj(pos_embedding).view(batch_size, self.num_heads, self.d_head, -1)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        # content_score = torch.einsum(self.make_attn_einsum_eq, (query + self.u_bias), key)
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        # pos_score = torch.einsum(self.make_attn_einsum_eq, (query + self.v_bias), pos_embedding)
        pos_score = self._compute_relative_positional_encoding(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim
        # print(score.shape)
        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)
        # print(score)
        # attn = F.softmax(score, -1)
        attn = self.Softmax_layer(score)
        attn = self.dropout(attn)

        # context = torch.matmul(attn, value).transpose(1, 2)
        context = torch.einsum(self.cal_attn_einsum_eq, attn, value)
        # context = context.contiguous().view(batch_size, -1, self.d_model)
        context = context.contiguous().view(batch_size, self.d_model, -1)

        return self.out_proj(context)

    def _compute_relative_positional_encoding(self, pos_score: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score


class FeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out


class LM_GNN(nn.Module):
    def __init__(
            self,
            lm_dim=1024,
            hidden_dim_1=256,
            hidden_dim_2=64,
            ab_freeze_layer_count=-3,
    ):
        super().__init__()
        self.ab_bert = BertModel.from_pretrained("train_0_165999")
        # freeze the ? layers
        _freeze_bert(self.ab_bert, freeze_bert=False, freeze_layer_count=ab_freeze_layer_count)
        # _freeze_bert(self.ab_bert, freeze_bert=True)

        self.ag_gnn1 = GCNConv(lm_dim, lm_dim)
        self.ag_gnn2 = GCNConv(lm_dim, lm_dim)
        self.ag_gnn3 = GCNConv(lm_dim, lm_dim)
        self.ag_bnorm1 = BatchNorm1d(lm_dim)
        self.ag_bnorm2 = BatchNorm1d(lm_dim)
        self.ag_bnorm3 = BatchNorm1d(lm_dim)
        self.cross_attn_norm1_ab = LayerNorm(d_model=lm_dim)
        self.cross_attn_norm1_ag = LayerNorm(d_model=lm_dim)
        self.cross_attn_norm2_ab = LayerNorm(d_model=lm_dim)
        self.cross_attn_norm2_ag = LayerNorm(d_model=lm_dim)
        self.self_attn_norm1_ab = LayerNorm(d_model=lm_dim)
        self.self_attn_norm1_ag = LayerNorm(d_model=lm_dim)
        self.self_attn_norm2_ab = LayerNorm(d_model=lm_dim)
        self.self_attn_norm2_ag = LayerNorm(d_model=lm_dim)
        self.cross_attn_fc_ab = FeedForward(d_model=lm_dim, hidden=hidden_dim_1)
        self.cross_attn_fc_ag = FeedForward(d_model=lm_dim, hidden=hidden_dim_1)
        self.self_attn_fc_ag = FeedForward(d_model=lm_dim, hidden=hidden_dim_1)
        self.self_attn_fc_ab = FeedForward(d_model=lm_dim, hidden=hidden_dim_1)
        self.cross_attn_layer_ab = RelativeMultiHeadAttention(d_model=lm_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag = RelativeMultiHeadAttention(d_model=lm_dim, num_heads=16, dropout_p=0.)
        self.self_attn_layer_ab = RelativeMultiHeadAttention(d_model=lm_dim, num_heads=16, dropout_p=0.)
        self.self_attn_layer_ag = RelativeMultiHeadAttention(d_model=lm_dim, num_heads=16, dropout_p=0.)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(nn.Linear(lm_dim, hidden_dim_1),
                                        Tanh(),
                                        nn.Dropout(0.),
                                        nn.Linear(hidden_dim_1, hidden_dim_2),
                                        Tanh(),
                                        nn.Dropout(0.),
                                        nn.Linear(hidden_dim_2, 1),
                                        nn.Sigmoid())

    def forward(self, input_ids_ab_v, attention_mask_ab_v, input_ids_ab_l, attention_mask_ab_l, ag_x, ag_edge_index,
                ag_ptr):
        ab_x_v = self.ab_bert(input_ids=input_ids_ab_v, attention_mask=attention_mask_ab_v).last_hidden_state
        ab_x_l = self.ab_bert(input_ids=input_ids_ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # print(ab_x_l.shape)

        ab_x = torch.cat([ab_x_v, ab_x_l], 1).permute(0, 2, 1)

        # GCN layer 1
        ag_h1 = self.ag_bnorm1(torch.tanh(self.ag_gnn1(ag_x, ag_edge_index)))
        ag_h1 = self.ag_bnorm2(torch.tanh(self.ag_gnn2(ag_h1, ag_edge_index)))
        # a big graph of mini-batch -> batch graphs
        ag_ptr = ag_ptr.tolist()
        ag_h1_2 = torch.split(ag_h1, (np.array(ag_ptr[1:]) - np.array(ag_ptr[:-1])).tolist(), 0)
        ag_h1_list = []
        for i in ag_h1_2:
            # padding = nn.ZeroPad2d(padding=(0, 0, 0, 512 - i.shape[0]))
            # ag_h1_list.append(padding(i))
            ag_h1_list.append(i)
        ag_h1 = torch.stack(ag_h1_list)
        ag_h1 = ag_h1.permute(0, 2, 1)

        # cross attention layer 1
        ag_h2 = self.cross_attn_layer_ag(query=ag_h1, key=ab_x, value=ab_x, pos_embedding=ab_x)
        ag_h2 = ag_h2.permute(0, 2, 1)
        ag_h1 = ag_h1.permute(0, 2, 1)
        ag_h2 = self.cross_attn_norm1_ag(ag_h1 + ag_h2)
        ag_h3 = self.cross_attn_fc_ag(ag_h2)
        ag_h3 = self.cross_attn_norm2_ag(ag_h3 + ag_h2)
        ag_h3 = ag_h3.permute(0, 2, 1)

        ab_h1 = self.cross_attn_layer_ab(query=ab_x, key=ag_h3, value=ag_h3, pos_embedding=ag_h3)
        ab_h1 = ab_h1.permute(0, 2, 1)
        ab_x = ab_x.permute(0, 2, 1)
        ab_h1 = self.cross_attn_norm1_ab(ab_h1 + ab_x)
        ab_h2 = self.cross_attn_fc_ab(ab_h1)
        ab_h2 = self.cross_attn_norm2_ab(ab_h1 + ab_h2)
        ab_h2 = ab_h2.permute(0, 2, 1)

        # self attention layer 1
        ag_h4 = self.self_attn_layer_ag(query=ag_h3, key=ag_h3, value=ag_h3, pos_embedding=ag_h3)
        ag_h3 = ag_h3.permute(0, 2, 1)
        ag_h4 = ag_h4.permute(0, 2, 1)
        ag_h4 = self.self_attn_norm1_ag(ag_h3 + ag_h4)
        ag_h5 = self.self_attn_fc_ag(ag_h4)
        ag_h5 = self.self_attn_norm2_ag(ag_h5 + ag_h4)
        ag_h5 = ag_h5.permute(0, 2, 1)

        ab_h3 = self.cross_attn_layer_ab(query=ab_h2, key=ag_h5, value=ag_h5, pos_embedding=ag_h5)
        ab_h3 = ab_h3.permute(0, 2, 1)
        ab_h2 = ab_h2.permute(0, 2, 1)
        ab_h3 = self.cross_attn_norm1_ab(ab_h3 + ab_h2)
        ab_h4 = self.cross_attn_fc_ab(ab_h3)
        ab_h4 = self.cross_attn_norm2_ab(ab_h3 + ab_h4)
        ab_h4 = ab_h4.permute(0, 2, 1)

        ab_h5 = self.self_attn_layer_ab(query=ab_h4, key=ab_h4, value=ab_h4, pos_embedding=ab_h4)
        ab_h5 = ab_h5.permute(0, 2, 1)
        ab_h4 = ab_h4.permute(0, 2, 1)
        ab_h5 = self.self_attn_norm1_ab(ab_h4 + ab_h5)
        ab_h6 = self.self_attn_fc_ab(ab_h5)
        ab_h6 = self.self_attn_norm2_ab(ab_h5 + ab_h6)
        ab_h6 = ab_h6.permute(0, 2, 1)

        ab_x = self.gmp(ab_h6).squeeze(2)
        # print(ab_x.shape)
        prediction = self.classifier(ab_x)
        # print(prediction.shape)

        return prediction


# ========================================================================================

class LM_ACGNN(nn.Module):
    def __init__(
            self,
            lm_dim=1024,
            ag_dim=30,
            hidden_dim_1=256,
            hidden_dim_2=64,
            ab_freeze_layer_count=-3,
    ):
        super().__init__()
        self.ab_bert = BertModel.from_pretrained("train_2_193999")
        # freeze the ? layers
        _freeze_bert(self.ab_bert, freeze_bert=False, freeze_layer_count=ab_freeze_layer_count)
        # _freeze_bert(self.ab_bert, freeze_bert=True)

        self.ab_acnn = nn.Sequential(
            Atrous_block(input_dim=lm_dim, output_dim=lm_dim),
            Atrous_block(input_dim=lm_dim, output_dim=lm_dim)
        )
        self.ag_acnn = nn.Sequential(
            Atrous_block(input_dim=1024, output_dim=lm_dim),
            Atrous_block(input_dim=lm_dim, output_dim=lm_dim)
        )

        self.ag_gnn1 = GCNConv(lm_dim, lm_dim)
        self.ag_gnn2 = GCNConv(lm_dim, lm_dim)
        self.ag_gnn3 = GCNConv(lm_dim, lm_dim)
        self.ag_bnorm1 = BatchNorm1d(lm_dim)
        self.ag_bnorm2 = BatchNorm1d(lm_dim)
        self.ag_bnorm3 = BatchNorm1d(lm_dim)
        self.cross_attn_norm1_ab = LayerNorm(d_model=lm_dim)
        self.cross_attn_norm1_ag = LayerNorm(d_model=lm_dim)
        self.cross_attn_norm2_ab = LayerNorm(d_model=lm_dim)
        self.cross_attn_norm2_ag = LayerNorm(d_model=lm_dim)
        self.self_attn_norm1_ab = LayerNorm(d_model=lm_dim)
        self.self_attn_norm1_ag = LayerNorm(d_model=lm_dim)
        self.self_attn_norm2_ab = LayerNorm(d_model=lm_dim)
        self.self_attn_norm2_ag = LayerNorm(d_model=lm_dim)
        # self.cross_attn_fc_ab = FeedForward(d_model=lm_dim, hidden=hidden_dim_1)
        # self.cross_attn_fc_ag = FeedForward(d_model=lm_dim, hidden=hidden_dim_1)
        # self.self_attn_fc_ag = FeedForward(d_model=lm_dim, hidden=hidden_dim_1)
        # self.self_attn_fc_ab = FeedForward(d_model=lm_dim, hidden=hidden_dim_1)
        self.cross_attn_fc_ab = Atrous_block(input_dim=lm_dim, output_dim=lm_dim, layer_num=2)
        self.cross_attn_fc_ag = Atrous_block(input_dim=lm_dim, output_dim=lm_dim, layer_num=2)
        self.self_attn_fc_ag = Atrous_block(input_dim=lm_dim, output_dim=lm_dim, layer_num=2)
        self.self_attn_fc_ab = Atrous_block(input_dim=lm_dim, output_dim=lm_dim, layer_num=2)

        self.cross_attn_layer_ab_hxl = RelativeMultiHeadAttention(d_model=lm_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab = RelativeMultiHeadAttention(d_model=lm_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag = RelativeMultiHeadAttention(d_model=lm_dim, num_heads=16, dropout_p=0.)
        self.self_attn_layer_ab = RelativeMultiHeadAttention(d_model=lm_dim, num_heads=16, dropout_p=0.)
        self.self_attn_layer_ag = RelativeMultiHeadAttention(d_model=lm_dim, num_heads=16, dropout_p=0.)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(nn.Linear(lm_dim, hidden_dim_1),
                                        Tanh(),
                                        nn.Dropout(0.1),
                                        nn.Linear(hidden_dim_1, hidden_dim_2),
                                        Tanh(),
                                        nn.Dropout(0.1),
                                        nn.Linear(hidden_dim_2, 1),
                                        nn.Sigmoid())

    def forward(self, input_ids_ab_v, attention_mask_ab_v, input_ids_ab_l, attention_mask_ab_l, ag_x, ag_edge_index,
                ag_ptr):
        ab_x_v = self.ab_bert(input_ids=input_ids_ab_v, attention_mask=attention_mask_ab_v).last_hidden_state

        # print(ab_x_l.shape)

        # ab_x = torch.cat([ab_x_v, ab_x_l], 1).permute(0, 2, 1)
        ab_x_v = ab_x_v.permute(0, 2, 1)
        ab_x_v = self.ab_acnn(ab_x_v)

        # ACNN

        #

        # Hevay x light
        ab_x_l = self.ab_bert(input_ids=input_ids_ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        ab_x_l = ab_x_l.permute(0, 2, 1)
        ab_x_l = self.ab_acnn(ab_x_l)
        ab_x = self.cross_attn_layer_ab_hxl(query=ab_x_v, key=ab_x_l, value=ab_x_l, pos_embedding=ab_x_l)

        # GCN layer 1
        ag_h1 = self.ag_bnorm1(torch.tanh(self.ag_gnn1(ag_x, ag_edge_index)))
        ag_h1 = self.ag_bnorm2(torch.tanh(self.ag_gnn2(ag_h1, ag_edge_index)))
        ag_h1 = self.ag_bnorm3(torch.tanh(self.ag_gnn3(ag_h1, ag_edge_index)))
        # a big graph of mini-batch -> batch graphs
        ag_ptr = ag_ptr.tolist()
        ag_h1_2 = torch.split(ag_h1, (np.array(ag_ptr[1:]) - np.array(ag_ptr[:-1])).tolist(), 0)
        ag_h1_list = []
        for i in ag_h1_2:
            ag_h1_list.append(i)
        ag_h1 = torch.stack(ag_h1_list)
        ag_h1 = ag_h1.permute(0, 2, 1)
        ag_h1 = self.ag_acnn(ag_h1)
        # cross attention layer 1
        ag_h2 = self.cross_attn_layer_ag(query=ag_h1, key=ab_x, value=ab_x, pos_embedding=ab_x)
        ag_h2 = ag_h2.permute(0, 2, 1)
        ag_h1 = ag_h1.permute(0, 2, 1)
        ag_h2 = self.cross_attn_norm1_ag(ag_h1 + ag_h2)
        ag_h2 = ag_h2.permute(0, 2, 1)
        ag_h3 = self.cross_attn_fc_ag(ag_h2)
        ag_h2 = ag_h2.permute(0, 2, 1)
        ag_h3 = ag_h3.permute(0, 2, 1)
        ag_h3 = self.cross_attn_norm2_ag(ag_h3 + ag_h2)
        ag_h3 = ag_h3.permute(0, 2, 1)

        ab_h1 = self.cross_attn_layer_ab(query=ab_x, key=ag_h3, value=ag_h3, pos_embedding=ag_h3)
        ab_h1 = ab_h1.permute(0, 2, 1)
        ab_x = ab_x.permute(0, 2, 1)
        ab_h1 = self.cross_attn_norm1_ab(ab_h1 + ab_x)
        ab_h1 = ab_h1.permute(0, 2, 1)
        ab_h2 = self.cross_attn_fc_ab(ab_h1)
        ab_h1 = ab_h1.permute(0, 2, 1)
        ab_h2 = ab_h2.permute(0, 2, 1)
        ab_h2 = self.cross_attn_norm2_ab(ab_h1 + ab_h2)
        ab_h2 = ab_h2.permute(0, 2, 1)

        # self attention layer 1
        ag_h4 = self.self_attn_layer_ag(query=ag_h3, key=ag_h3, value=ag_h3, pos_embedding=ag_h3)
        ag_h3 = ag_h3.permute(0, 2, 1)
        ag_h4 = ag_h4.permute(0, 2, 1)
        ag_h4 = self.self_attn_norm1_ag(ag_h3 + ag_h4)
        ag_h4 = ag_h4.permute(0, 2, 1)
        ag_h5 = self.self_attn_fc_ag(ag_h4)
        ag_h4 = ag_h4.permute(0, 2, 1)
        ag_h5 = ag_h5.permute(0, 2, 1)
        ag_h5 = self.self_attn_norm2_ag(ag_h5 + ag_h4)
        ag_h5 = ag_h5.permute(0, 2, 1)

        ab_h3 = self.cross_attn_layer_ab(query=ab_h2, key=ag_h5, value=ag_h5, pos_embedding=ag_h5)
        ab_h3 = ab_h3.permute(0, 2, 1)
        ab_h2 = ab_h2.permute(0, 2, 1)
        ab_h3 = self.cross_attn_norm1_ab(ab_h3 + ab_h2)
        ab_h3 = ab_h3.permute(0, 2, 1)
        ab_h4 = self.cross_attn_fc_ab(ab_h3)
        ab_h3 = ab_h3.permute(0, 2, 1)
        ab_h4 = ab_h4.permute(0, 2, 1)
        ab_h4 = self.cross_attn_norm2_ab(ab_h3 + ab_h4)
        ab_h4 = ab_h4.permute(0, 2, 1)

        ab_h5 = self.self_attn_layer_ab(query=ab_h4, key=ab_h4, value=ab_h4, pos_embedding=ab_h4)
        ab_h5 = ab_h5.permute(0, 2, 1)
        ab_h4 = ab_h4.permute(0, 2, 1)
        ab_h5 = self.self_attn_norm1_ab(ab_h4 + ab_h5)
        ab_h5 = ab_h5.permute(0, 2, 1)
        ab_h6 = self.self_attn_fc_ab(ab_h5)
        ab_h5 = ab_h5.permute(0, 2, 1)
        ab_h6 = ab_h6.permute(0, 2, 1)
        ab_h6 = self.self_attn_norm2_ab(ab_h5 + ab_h6)
        ab_h6 = ab_h6.permute(0, 2, 1)

        ab_x = self.gmp(ab_h6).squeeze(2)
        # print(ab_x.shape)
        prediction = self.classifier(ab_x)
        # print(prediction.shape)

        return prediction


class Atrous_block(torch.nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024, layer_num=3, normalize=False, ndims=1):
        super(Atrous_block, self).__init__()
        self.layer_num = layer_num
        self.dimension = ndims
        self.normalize = normalize
        self.dim_trans = (input_dim != output_dim)
        self.identity = torch.nn.Identity()
        Norm = getattr(torch.nn, 'InstanceNorm%dd' % self.dimension)  # nn.InstanceNorm2d
        Conv = getattr(torch.nn, 'Conv%dd' % self.dimension)
        # self.act = torch.nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        self.act = torch.nn.ReLU()
        self.norm_layer = torch.nn.ModuleList(
            [Norm(self.dimension, affine=False) if normalize else self.identity for _ in range(layer_num)])

        # self.input_layer = Conv(input_dim, output_dim, kernel_size=3, padding=1)
        if self.dim_trans:
            self.input_layer = torch.nn.Sequential(
                Conv(input_dim, output_dim, kernel_size=3, padding=1),
                Norm(self.dimension, affine=False),
                # self.act,
            )
        else:
            self.input_layer = self.identity
        self.conv_layers = torch.nn.ModuleList(
            [Conv(output_dim, output_dim, kernel_size=3, dilation=3 ** i, padding=3 ** i) for i in range(layer_num)])

    def forward(self, x):
        x = self.input_layer(x)  # if self.dim_trans else x
        y = self.identity(x)
        for i in range(self.layer_num):
            y = self.norm_layer[i](self.conv_layers[i](self.act(y)))
        return self.act(y + x)

class Atrous_block_leaky(torch.nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024, layer_num=3, normalize=False, ndims=1):
        super(Atrous_block_leaky, self).__init__()
        self.layer_num = layer_num
        self.dimension = ndims
        self.normalize = normalize
        self.dim_trans = (input_dim != output_dim)
        self.identity = torch.nn.Identity()
        Norm = getattr(torch.nn, 'InstanceNorm%dd' % self.dimension)  # nn.InstanceNorm2d
        Conv = getattr(torch.nn, 'Conv%dd' % self.dimension)
        self.act = torch.nn.LeakyReLU(negative_slope=1e-2)
        # self.act = torch.nn.ReLU()
        self.norm_layer = torch.nn.ModuleList(
            [Norm(self.dimension, affine=False) if normalize else self.identity for _ in range(layer_num)])

        # self.input_layer = Conv(input_dim, output_dim, kernel_size=3, padding=1)
        if self.dim_trans:
            self.input_layer = torch.nn.Sequential(
                Conv(input_dim, output_dim, kernel_size=3, padding=1),
                Norm(self.dimension, affine=False),
                # self.act,
            )
        else:
            self.input_layer = self.identity
        self.conv_layers = torch.nn.ModuleList(
            [Conv(output_dim, output_dim, kernel_size=3, dilation=3 ** i, padding=3 ** i) for i in range(layer_num)])

    def forward(self, x):
        x = self.input_layer(x)  # if self.dim_trans else x
        y = self.identity(x)
        for i in range(self.layer_num):
            y = self.norm_layer[i](self.conv_layers[i](self.act(y)))
        return self.act(y + x)

class XBCR_ACNN(torch.nn.Module):

    def __init__(self, node_attr_dim=1024, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=3, freeze_bert=False, ab_freeze_layer_count=None, extra_dense=False):
        super(XBCR_ACNN, self).__init__()
        # self.ab_bert = BertModel.from_pretrained("train_2_193999")
        self.ab_bert = BertModel.from_pretrained("prot_bert")
        # freeze the ? layers
        if ab_freeze_layer_count is not None:
            _freeze_bert(self.ab_bert, freeze_bert=freeze_bert, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()
        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)

        self.dense_concat = torch.nn.Linear(3 * self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.classifier = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

    def get_variables(self,lp=1):
        self.device=next(self.dense_concat.parameters()).device
        total_regul=torch.tensor(0,dtype=torch.float,device=self.device)
        total_param_num = 1
        for param in self.parameters():
            total_regul += torch.norm(param, lp) ** 1
            total_param_num += param.numel()
        cls_x_regul,cls_y_regul=torch.tensor(0,dtype=torch.float,device=self.device),torch.tensor(0,dtype=torch.float,device=self.device)
        cls_x_param_num, cls_y_param_num = 1, 1
        # if self.extra_dense:
        #     cls_x=[
        #         self.dense_concat,
        #         self.res_dense_block1,
        #         self.res_dense_block2,
        #         # self.classifier,
        #     ]
        #     for model in cls_x:
        #         # print(model)
        #         for param in model.parameters():
        #             cls_x_regul += torch.norm(param, lp) ** 1
        #             cls_x_param_num += param.numel()

        # if self.extra_classifier:
        #     cls_y=[
        #         self.dense_concat_ab,
        #         self.res_dense_block3,
        #         self.res_dense_block4,
        #         # self.classifier_ar,
        #     ]
        #     for model in cls_y:
        #         for param in model.parameters():
        #             cls_y_regul += torch.norm(param, lp) ** 2
        #             cls_y_param_num += param.numel()
        # return [0.01*cls_x_regul/cls_x_param_num,0.01*cls_y_regul/cls_y_param_num]
        return [0.01 * total_regul / total_param_num,0.01*cls_x_regul/cls_x_param_num]

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None):
        ab_x = self.ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
        ab_l = self.ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])
        ab_x = torch.permute(ab_x, [0, 2, 1])
        ab_l = torch.permute(ab_l, [0, 2, 1])

        ag_h1 = self.ag_gnn1(ag_x)
        ag_h1 = self.ag_gnn2(ag_h1)

        # antibody acnn
        ab_h1 = self.ab_gnn1(ab_x)
        ab_h1 = self.ab_gnn2(ab_h1)

        ab_l1 = self.ab_gnn1(ab_l)
        ab_l1 = self.ab_gnn2(ab_l1)

        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h1, kernel_size=ag_h1.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h1, kernel_size=ab_h1.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l1, kernel_size=ab_l1.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
        x = self.dense_concat(x)
        x = self.classifier(x + self.res_dense_block(x))
        x = self.sigmoid(x)

        return [x]



class XBCR_ACNN_woBERT(torch.nn.Module):

    def __init__(self, node_attr_dim=20, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=3, freeze_bert=False, ab_freeze_layer_count=None, extra_dense=False):
        super(XBCR_ACNN_woBERT, self).__init__()
        # self.ab_bert = BertModel.from_pretrained("train_2_193999")
        # self.ab_bert = BertModel.from_pretrained("prot_bert")
        # freeze the ? layers
        # if ab_freeze_layer_count is not None:
        #     _freeze_bert(self.ab_bert, freeze_bert=freeze_bert, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()
        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)

        self.dense_concat = torch.nn.Linear(3 * self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            # torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.classifier = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

    def get_variables(self,lp=1):
        self.device=next(self.dense_concat.parameters()).device
        total_regul=torch.tensor(0,dtype=torch.float,device=self.device)
        total_param_num = 1
        for param in self.parameters():
            total_regul += torch.norm(param, lp) ** 1
            total_param_num += param.numel()
        cls_x_regul,cls_y_regul=torch.tensor(0,dtype=torch.float,device=self.device),torch.tensor(0,dtype=torch.float,device=self.device)
        cls_x_param_num, cls_y_param_num = 1, 1
        # if self.extra_dense:
        #     cls_x=[
        #         self.dense_concat,
        #         self.res_dense_block1,
        #         self.res_dense_block2,
        #         # self.classifier,
        #     ]
        #     for model in cls_x:
        #         # print(model)
        #         for param in model.parameters():
        #             cls_x_regul += torch.norm(param, lp) ** 1
        #             cls_x_param_num += param.numel()

        # if self.extra_classifier:
        #     cls_y=[
        #         self.dense_concat_ab,
        #         self.res_dense_block3,
        #         self.res_dense_block4,
        #         # self.classifier_ar,
        #     ]
        #     for model in cls_y:
        #         for param in model.parameters():
        #             cls_y_regul += torch.norm(param, lp) ** 2
        #             cls_y_param_num += param.numel()
        # return [0.01*cls_x_regul/cls_x_param_num,0.01*cls_y_regul/cls_y_param_num]
        return [0.01 * total_regul / total_param_num,0.01*cls_x_regul/cls_x_param_num]

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None):
        # ab_x = self.ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
        # ab_l = self.ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])
        ab_x = torch.permute(ab_x, [0, 2, 1])
        ab_l = torch.permute(ab_l, [0, 2, 1])

        ag_h1 = self.ag_gnn1(ag_x)
        ag_h1 = self.ag_gnn2(ag_h1)

        # antibody acnn
        ab_h1 = self.ab_gnn1(ab_x)
        ab_h1 = self.ab_gnn2(ab_h1)

        ab_l1 = self.ab_gnn1(ab_l)
        ab_l1 = self.ab_gnn2(ab_l1)

        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h1, kernel_size=ag_h1.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h1, kernel_size=ab_h1.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l1, kernel_size=ab_l1.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
        x = self.dense_concat(x)
        x = self.classifier(x + self.res_dense_block(x))
        x = self.sigmoid(x)

        return [x]

class XBCR_ACNN_woBERT_leaky(torch.nn.Module):

    def __init__(self, node_attr_dim=20, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=True, dropout=0.,
                 heads=4, block_num=5, freeze_bert=False, ab_freeze_layer_count=None, extra_dense=False):
        super(XBCR_ACNN_woBERT_leaky, self).__init__()
        # self.ab_bert = BertModel.from_pretrained("train_2_193999")
        # self.ab_bert = BertModel.from_pretrained("prot_bert")
        # freeze the ? layers
        # if ab_freeze_layer_count is not None:
        #     _freeze_bert(self.ab_bert, freeze_bert=freeze_bert, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block_leaky(20, self.hidden_dim)
        atrous_blocks = [Atrous_block_leaky(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()
        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
        else:
            self.ab_gnn1 = Atrous_block_leaky(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block_leaky(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)

        self.dense_concat = torch.nn.Linear(3 * self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.classifier = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

    def get_variables(self,lp=1):
        self.device=next(self.dense_concat.parameters()).device
        total_regul=torch.tensor(0,dtype=torch.float,device=self.device)
        total_param_num = 1
        for param in self.parameters():
            total_regul += torch.norm(param, lp) ** 1
            total_param_num += param.numel()
        cls_x_regul,cls_y_regul=torch.tensor(0,dtype=torch.float,device=self.device),torch.tensor(0,dtype=torch.float,device=self.device)
        cls_x_param_num, cls_y_param_num = 1, 1
        # if self.extra_dense:
        #     cls_x=[
        #         self.dense_concat,
        #         self.res_dense_block1,
        #         self.res_dense_block2,
        #         # self.classifier,
        #     ]
        #     for model in cls_x:
        #         # print(model)
        #         for param in model.parameters():
        #             cls_x_regul += torch.norm(param, lp) ** 1
        #             cls_x_param_num += param.numel()

        # if self.extra_classifier:
        #     cls_y=[
        #         self.dense_concat_ab,
        #         self.res_dense_block3,
        #         self.res_dense_block4,
        #         # self.classifier_ar,
        #     ]
        #     for model in cls_y:
        #         for param in model.parameters():
        #             cls_y_regul += torch.norm(param, lp) ** 2
        #             cls_y_param_num += param.numel()
        # return [0.01*cls_x_regul/cls_x_param_num,0.01*cls_y_regul/cls_y_param_num]
        return [0.01 * total_regul / total_param_num,0.01*cls_x_regul/cls_x_param_num]

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None):
        # ab_x = self.ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
        # ab_l = self.ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])
        ab_x = torch.permute(ab_x, [0, 2, 1])
        ab_l = torch.permute(ab_l, [0, 2, 1])

        ag_h1 = self.ag_gnn1(ag_x)
        ag_h1 = self.ag_gnn2(ag_h1)

        # antibody acnn
        ab_h1 = self.ab_gnn1(ab_x)
        ab_h1 = self.ab_gnn2(ab_h1)

        ab_l1 = self.ab_gnn1(ab_l)
        ab_l1 = self.ab_gnn2(ab_l1)

        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h1, kernel_size=ag_h1.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h1, kernel_size=ab_h1.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l1, kernel_size=ab_l1.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
        x = self.dense_concat(x)
        x = self.classifier(x + self.res_dense_block(x))
        x = self.sigmoid(x)

        return [x]

class XBCR_ACNN_woBERT_relu(torch.nn.Module):

    def __init__(self, node_attr_dim=20, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=True, dropout=0.,
                 heads=4, block_num=5, freeze_bert=False, ab_freeze_layer_count=None, extra_dense=False):
        super(XBCR_ACNN_woBERT_relu, self).__init__()
        # self.ab_bert = BertModel.from_pretrained("train_2_193999")
        # self.ab_bert = BertModel.from_pretrained("prot_bert")
        # freeze the ? layers
        # if ab_freeze_layer_count is not None:
        #     _freeze_bert(self.ab_bert, freeze_bert=freeze_bert, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        # self.relu = torch.nn.LeakyReLU()
        self.relu = torch.nn.ReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()
        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)

        self.dense_concat = torch.nn.Linear(3 * self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.classifier = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

    def get_variables(self,lp=1):
        self.device=next(self.dense_concat.parameters()).device
        total_regul=torch.tensor(0,dtype=torch.float,device=self.device)
        total_param_num = 1
        for param in self.parameters():
            total_regul += torch.norm(param, lp) ** 1
            total_param_num += param.numel()
        cls_x_regul,cls_y_regul=torch.tensor(0,dtype=torch.float,device=self.device),torch.tensor(0,dtype=torch.float,device=self.device)
        cls_x_param_num, cls_y_param_num = 1, 1
        # if self.extra_dense:
        #     cls_x=[
        #         self.dense_concat,
        #         self.res_dense_block1,
        #         self.res_dense_block2,
        #         # self.classifier,
        #     ]
        #     for model in cls_x:
        #         # print(model)
        #         for param in model.parameters():
        #             cls_x_regul += torch.norm(param, lp) ** 1
        #             cls_x_param_num += param.numel()

        # if self.extra_classifier:
        #     cls_y=[
        #         self.dense_concat_ab,
        #         self.res_dense_block3,
        #         self.res_dense_block4,
        #         # self.classifier_ar,
        #     ]
        #     for model in cls_y:
        #         for param in model.parameters():
        #             cls_y_regul += torch.norm(param, lp) ** 2
        #             cls_y_param_num += param.numel()
        # return [0.01*cls_x_regul/cls_x_param_num,0.01*cls_y_regul/cls_y_param_num]
        return [0.01 * total_regul / total_param_num,0.01*cls_x_regul/cls_x_param_num]

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None):
        # ab_x = self.ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
        # ab_l = self.ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])
        ab_x = torch.permute(ab_x, [0, 2, 1])
        ab_l = torch.permute(ab_l, [0, 2, 1])

        ag_h1 = self.ag_gnn1(ag_x)
        ag_h1 = self.ag_gnn2(ag_h1)

        # antibody acnn
        ab_h1 = self.ab_gnn1(ab_x)
        ab_h1 = self.ab_gnn2(ab_h1)

        ab_l1 = self.ab_gnn1(ab_l)
        ab_l1 = self.ab_gnn2(ab_l1)

        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h1, kernel_size=ag_h1.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h1, kernel_size=ab_h1.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l1, kernel_size=ab_l1.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
        x = self.dense_concat(x)
        x = self.classifier(x + self.res_dense_block(x))
        x = self.sigmoid(x)

        return [x]


class CrossMultiHeadAttention(torch.nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout
    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, feat, seq): Tensor containing query vector
        - **key** (batch, feat, seq): Tensor containing key vector
        - **value** (batch, feat, seq): Tensor containing value vector
        - **pos_embedding** (batch, feat, seq): Positional embedding tensor
        - **mask** (batch, 1, seq2) or (batch, seq1, seq2): Tensor containing indices to be masked
    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """

    def __init__(
            self,
            d_model: int = 1024,
            num_heads: int = 16,
            dropout_p: float = 0.,
            ndims: int = 1,
    ):
        super(CrossMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        # proj_layer=getattr(torch.nn,'Linear')
        # self.query_proj = proj_layer(d_model, d_model)
        # self.key_proj = proj_layer(d_model, d_model)
        # self.value_proj = proj_layer(d_model, d_model)
        # self.pos_proj = proj_layer(d_model, d_model, bias=False)
        # self.out_proj = proj_layer(d_model, d_model)
        proj_layer = getattr(torch.nn, 'Conv%dd' % ndims)
        self.query_proj = proj_layer(d_model, d_model, 1)
        self.key_proj = proj_layer(d_model, d_model, 1)
        self.value_proj = proj_layer(d_model, d_model, 1)
        self.pos_proj = proj_layer(d_model, d_model, 1, bias=False)
        self.out_proj = proj_layer(d_model, d_model, 1)

        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.u_bias = torch.nn.Parameter(torch.Tensor(self.num_heads, self.d_head, 1))
        self.v_bias = torch.nn.Parameter(torch.Tensor(self.num_heads, self.d_head, 1))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)
        self.Softmax_layer = torch.nn.Softmax(dim=-1)
        self.make_attn_einsum_eq = 'b h k i, b h k j -> b h i j'
        self.cal_attn_einsum_eq = 'b h i j, b h k j -> b h k i'

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            pos_embedding: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = value.size(0)

        # query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        query = self.query_proj(query).view(batch_size, self.num_heads, self.d_head, -1)
        # key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        key = self.key_proj(key).view(batch_size, self.num_heads, self.d_head, -1)  # .permute(0, 2, 1, 3)
        # value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, self.num_heads, self.d_head, -1)  # .permute(0, 2, 1, 3)
        # pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, self.num_heads, self.d_head, -1)

        # content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        content_score = torch.einsum(self.make_attn_einsum_eq, (query + self.u_bias), key)
        # pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = torch.einsum(self.make_attn_einsum_eq, (query + self.v_bias), pos_embedding)
        pos_score = self._compute_relative_positional_encoding(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim
        # print(score.shape)
        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)
        # print(score)
        # attn = F.softmax(score, -1)
        attn = self.Softmax_layer(score)
        attn = self.dropout(attn)

        # context = torch.matmul(attn, value).transpose(1, 2)
        context = torch.einsum(self.cal_attn_einsum_eq, attn, value)
        # context = context.contiguous().view(batch_size, -1, self.d_model)
        context = context.contiguous().view(batch_size, self.d_model, -1)

        return self.out_proj(context)

    def _compute_relative_positional_encoding(self, pos_score: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score

class EvolutMultiHeadAttention(torch.nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout
    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, feat, seq): Tensor containing query vector
        - **key** (batch, feat, seq): Tensor containing key vector
        - **value** (batch, feat, seq): Tensor containing value vector
        - **pos_embedding** (batch, feat, seq): Positional embedding tensor
        - **mask** (batch, 1, seq2) or (batch, seq1, seq2): Tensor containing indices to be masked
    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """

    def __init__(
            self,
            d_model: int = 1024,
            num_heads: int = 16,
            dropout_p: float = 0.,
            ndims: int = 1,
    ):
        super(EvolutMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        # proj_layer=getattr(torch.nn,'Linear')
        # self.query_proj = proj_layer(d_model, d_model)
        # self.key_proj = proj_layer(d_model, d_model)
        # self.value_proj = proj_layer(d_model, d_model)
        # self.pos_proj = proj_layer(d_model, d_model, bias=False)
        # self.out_proj = proj_layer(d_model, d_model)
        proj_layer = getattr(torch.nn, 'Conv%dd' % ndims)
        self.query_proj = proj_layer(d_model, d_model, 1)
        self.key_proj = proj_layer(d_model, d_model, 1)
        self.value_proj = proj_layer(d_model, d_model, 1)
        # self.pos_proj = proj_layer(d_model, d_model, 1, bias=False)
        self.out_proj = proj_layer(d_model, d_model, 1)

        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.u_bias = torch.nn.Parameter(torch.Tensor(self.num_heads, self.d_head, 1))
        self.v_bias = torch.nn.Parameter(torch.Tensor(self.num_heads, self.d_head, 1))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)
        self.Softmax_layer = torch.nn.Softmax(dim=-1)
        self.make_attn_einsum_eq = 'b h k i, b h k j -> b h i j'
        self.cal_attn_einsum_eq = 'b h i j, b h k j -> b h k i'
        self.evolut_fusion = torch.nn.Sequential(torch.nn.Conv2d(self.num_heads*2,self.num_heads*4,kernel_size=3,padding=1,stride=1),
                                                 torch.nn.LeakyReLU(negative_slope=1e-1,inplace=True),
                                                 torch.nn.Conv2d(self.num_heads*4,self.num_heads,kernel_size=3,padding=1,stride=1))

        self.evolut_fusion = torch.nn.Conv2d(self.num_heads*2,self.num_heads,kernel_size=3,padding=1,stride=1)
    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            pos_embedding: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            edge_weights: Optional[torch.Tensor] = None,
    ) -> (torch.Tensor,torch.Tensor):
        batch_size = value.size(0)

        # query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        query = self.query_proj(query).view(batch_size, self.num_heads, self.d_head, -1)
        # key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        key = self.key_proj(key).view(batch_size, self.num_heads, self.d_head, -1)  # .permute(0, 2, 1, 3)
        # value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, self.num_heads, self.d_head, -1)  # .permute(0, 2, 1, 3)
        # pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)
        # pos_embedding = self.pos_proj(pos_embedding).view(batch_size, self.num_heads, self.d_head, -1)

        # content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        content_score = torch.einsum(self.make_attn_einsum_eq, (query + self.u_bias), key)
        # pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        # pos_score = torch.einsum(self.make_attn_einsum_eq, (query + self.v_bias), pos_embedding)
        # pos_score = self._compute_relative_positional_encoding(pos_score)

        # score = (content_score + pos_score) / self.sqrt_dim
        score = (content_score) / self.sqrt_dim
        if edge_weights is not None:
            score = self.evolut_fusion(torch.cat([score,edge_weights],1))
        # print(score.shape)
        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)
        # print(score)
        # attn = F.softmax(score, -1)
        attn = self.Softmax_layer(score)
        attn = self.dropout(attn)

        # context = torch.matmul(attn, value).transpose(1, 2)
        context = torch.einsum(self.cal_attn_einsum_eq, attn, value)
        # context = context.contiguous().view(batch_size, -1, self.d_model)
        context = context.contiguous().view(batch_size, self.d_model, -1)

        return self.out_proj(context), score
        # return self.out_proj(context), attn

    def _compute_relative_positional_encoding(self, pos_score: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score

class XBCR_ACXN(torch.nn.Module):

    def __init__(self, node_attr_dim=1024, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=2, ab_freeze_layer_count=None):
        super(XBCR_ACXN, self).__init__()
        self.ab_bert = BertModel.from_pretrained("train_2_193999")
        # freeze the ? layers
        if ab_freeze_layer_count is not None:
            _freeze_bert(self.ab_bert, freeze_bert=False, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn3 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()

        self.cross_attn_layer_ab1 = CrossMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag1 = CrossMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab2 = CrossMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag2 = CrossMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)

        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
            self.ab_gnn3 = self.ag_gnn3
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn3 = torch.nn.Sequential(*atrous_blocks)

        self.dense_concat = torch.nn.Linear(3 * self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.classifier = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(3 * self.hidden_dim, self.h1_dim),
        #     self.relu,
        #     torch.nn.Linear(self.h1_dim ,self.h2_dim),
        #     self.relu,
        #     torch.nn.Dropout(dropout),
        #     torch.nn.Linear(self.h2_dim, 1))
        #

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None):
        ab_x = self.ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
        ab_l = self.ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])
        ab_x = torch.permute(ab_x, [0, 2, 1])
        ab_l = torch.permute(ab_l, [0, 2, 1])

        # antibody acnn x cross attention
        ab_h1 = self.ab_gnn1(ab_x)
        ab_l1 = self.ab_gnn1(ab_l)
        ag_h1 = self.ag_gnn1(ag_x)

        ab_h2 = ab_h1 + self.cross_attn_layer_ag1(query=ab_h1, key=ag_h1, value=ag_h1,
                                                  pos_embedding=ag_h1) + self.cross_attn_layer_ab1(query=ab_h1,
                                                                                                   key=ab_l1,
                                                                                                   value=ab_l1,
                                                                                                   pos_embedding=ab_l1)
        ab_l2 = ab_l1 + self.cross_attn_layer_ab1(query=ab_l1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ag_h2 = ag_h1 + self.cross_attn_layer_ag1(query=ag_h1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)

        ab_h2 = self.ab_gnn2(ab_h2)
        ab_l2 = self.ab_gnn2(ab_l2)
        ag_h2 = self.ag_gnn2(ag_h2)

        ab_h3 = ab_h2 + self.cross_attn_layer_ag2(query=ab_h2, key=ag_h2, value=ag_h2,
                                                  pos_embedding=ag_h2) + self.cross_attn_layer_ab2(query=ab_h2,
                                                                                                   key=ab_l2,
                                                                                                   value=ab_l2,
                                                                                                   pos_embedding=ab_l2)
        ab_l3 = ab_l2 + self.cross_attn_layer_ab2(query=ab_l2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2)
        ag_h3 = ag_h2 + self.cross_attn_layer_ag2(query=ag_h2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2)

        ab_h3 = self.ab_gnn3(ab_h3)
        ab_l3 = self.ab_gnn3(ab_l3)
        ag_h3 = self.ag_gnn3(ag_h3)

        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h3, kernel_size=ag_h3.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h3, kernel_size=ab_h3.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l3, kernel_size=ab_l3.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
        # x = self.classifier(x)

        x = self.dense_concat(x)
        x = self.classifier(x + self.res_dense_block(x))
        x = self.sigmoid(x)

        return x

class XBCR_ACEXN(torch.nn.Module):

    def __init__(self, node_attr_dim=1024, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=2, ab_freeze_layer_count=None,output_attn_score=False):
        super(XBCR_ACEXN, self).__init__()
        self.output_attn_score=output_attn_score
        self.ab_bert = BertModel.from_pretrained("train_2_193999")
        # freeze the ? layers
        if ab_freeze_layer_count is not None:
            _freeze_bert(self.ab_bert, freeze_bert=False, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn3 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()

        self.cross_attn_layer_abh2l1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abl2h1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag2ab1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab2ag1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abh2l2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abl2h2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag2ab2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab2ag2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)

        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
            self.ab_gnn3 = self.ag_gnn3
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn3 = torch.nn.Sequential(*atrous_blocks)

        self.dense_concat = torch.nn.Linear(3 * self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.classifier = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(3 * self.hidden_dim, self.h1_dim),
        #     self.relu,
        #     torch.nn.Linear(self.h1_dim ,self.h2_dim),
        #     self.relu,
        #     torch.nn.Dropout(dropout),
        #     torch.nn.Linear(self.h2_dim, 1))
        #

    def get_variables(self,lp=1):
        self.device=next(self.dense_concat.parameters()).device
        total_regul=torch.tensor(0,dtype=torch.float,device=self.device)
        total_param_num = 1
        for param in self.parameters():
            total_regul += torch.norm(param, lp) ** 1
            total_param_num += param.numel()
        cls_x_regul,cls_y_regul=torch.tensor(0,dtype=torch.float,device=self.device),torch.tensor(0,dtype=torch.float,device=self.device)
        cls_x_param_num, cls_y_param_num = 1, 1
        if self.extra_dense:
            cls_x=[
                self.dense_concat,
                self.res_dense_block1,
                self.res_dense_block2,
                # self.classifier,
            ]
            for model in cls_x:
                # print(model)
                for param in model.parameters():
                    cls_x_regul += torch.norm(param, lp) ** 1
                    cls_x_param_num += param.numel()

        # if self.extra_classifier:
        #     cls_y=[
        #         self.dense_concat_ab,
        #         self.res_dense_block3,
        #         self.res_dense_block4,
        #         # self.classifier_ar,
        #     ]
        #     for model in cls_y:
        #         for param in model.parameters():
        #             cls_y_regul += torch.norm(param, lp) ** 2
        #             cls_y_param_num += param.numel()
        # return [0.01*cls_x_regul/cls_x_param_num,0.01*cls_y_regul/cls_y_param_num]
        return [0.01 * total_regul / total_param_num,0.01*cls_x_regul/cls_x_param_num]

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None,attn_score=None):
        # [ag2ab_s,abl2abh_s,abh2abl_s,ab2ag_s]=attn_score
        ab_x = self.ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
        ab_l = self.ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])
        ab_x = torch.permute(ab_x, [0, 2, 1])
        ab_l = torch.permute(ab_l, [0, 2, 1])

        # antibody acnn x cross attention
        ab_h1 = self.ab_gnn1(ab_x)
        ab_l1 = self.ab_gnn1(ab_l)
        ag_h1 = self.ag_gnn1(ag_x)

        ag2ab_x,ag2ab_s = self.cross_attn_layer_ag2ab1(query=ab_h1, key=ag_h1, value=ag_h1, pos_embedding=ag_h1)
        abl2abh_x,abl2abh_s = self.cross_attn_layer_abl2h1(query=ab_h1, key=ab_l1, value=ab_l1,pos_embedding=ab_l1)
        ab_h2 = ab_h1 + ag2ab_x + abl2abh_x
        abh2abl_x,abh2abl_s = self.cross_attn_layer_abh2l1(query=ab_l1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ab_l2 = ab_l1 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag1(query=ag_h1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ag_h2 = ag_h1 + ab2ag_x

        ab_h2 = self.ab_gnn2(ab_h2)
        ab_l2 = self.ab_gnn2(ab_l2)
        ag_h2 = self.ag_gnn2(ag_h2)

        ag2ab_x, ag2ab_s = self.cross_attn_layer_ag2ab2(query=ab_h2, key=ag_h2, value=ag_h2, pos_embedding=ag_h2, edge_weights=ag2ab_s)
        abl2abh_x, abl2abh_s = self.cross_attn_layer_abl2h2(query=ab_h2, key=ab_l2, value=ab_l2, pos_embedding=ab_l2, edge_weights=abl2abh_s)
        ab_h3 = ab_h2 + ag2ab_x + abl2abh_x
        abh2abl_x, abh2abl_s = self.cross_attn_layer_abh2l2(query=ab_l2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=abh2abl_s)
        ab_l3 = ab_l2 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag2(query=ag_h2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=ab2ag_s)
        ag_h3 = ag_h2 + ab2ag_x

        ab_h3 = self.ab_gnn3(ab_h3)
        ab_l3 = self.ab_gnn3(ab_l3)
        ag_h3 = self.ag_gnn3(ag_h3)

        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h3, kernel_size=ag_h3.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h3, kernel_size=ab_h3.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l3, kernel_size=ab_l3.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
        # x = self.classifier(x)

        x = self.dense_concat(x)
        x = self.classifier(x + self.res_dense_block(x))
        x = self.sigmoid(x)
        if self.output_attn_score:
            return x,[torch.sum(ag2ab_s,dim=1) * torch.sum(torch.permute(ab2ag_s,(0,1,3,2)),dim=1),torch.sum(torch.permute(abh2abl_s,(0,1,3,2)),dim=1) * torch.sum(abl2abh_s,dim=1)]
        else:
            return x

class XBCR_ACEXN_ED(torch.nn.Module):

    def __init__(self, node_attr_dim=1024, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=2, ab_freeze_layer_count=None,output_attn_score=False,extra_dense=True):
        super(XBCR_ACEXN_ED, self).__init__()
        self.output_attn_score=output_attn_score
        self.extra_dense=extra_dense
        self.ab_bert = BertModel.from_pretrained("train_2_193999")
        # freeze the ? layers
        if ab_freeze_layer_count is not None:
            _freeze_bert(self.ab_bert, freeze_bert=False, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn3 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()

        self.cross_attn_layer_abh2l1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abl2h1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag2ab1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab2ag1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abh2l2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abl2h2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag2ab2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab2ag2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)

        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
            self.ab_gnn3 = self.ag_gnn3
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn3 = torch.nn.Sequential(*atrous_blocks)

        self.dense_concat = torch.nn.Linear(3 * self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        if self.extra_dense:
            self.res_dense_block1 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
            self.res_dense_block2 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.classifier = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(3 * self.hidden_dim, self.h1_dim),
        #     self.relu,
        #     torch.nn.Linear(self.h1_dim ,self.h2_dim),
        #     self.relu,
        #     torch.nn.Dropout(dropout),
        #     torch.nn.Linear(self.h2_dim, 1))
        #

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None,attn_score=None):
        # [ag2ab_s,abl2abh_s,abh2abl_s,ab2ag_s]=attn_score
        ab_x = self.ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
        ab_l = self.ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])
        ab_x = torch.permute(ab_x, [0, 2, 1])
        ab_l = torch.permute(ab_l, [0, 2, 1])

        # antibody acnn x cross attention
        ab_h1 = self.ab_gnn1(ab_x)
        ab_l1 = self.ab_gnn1(ab_l)
        ag_h1 = self.ag_gnn1(ag_x)

        ag2ab_x,ag2ab_s = self.cross_attn_layer_ag2ab1(query=ab_h1, key=ag_h1, value=ag_h1, pos_embedding=ag_h1)
        abl2abh_x,abl2abh_s = self.cross_attn_layer_abl2h1(query=ab_h1, key=ab_l1, value=ab_l1,pos_embedding=ab_l1)
        ab_h2 = ab_h1 + ag2ab_x + abl2abh_x
        abh2abl_x,abh2abl_s = self.cross_attn_layer_abh2l1(query=ab_l1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ab_l2 = ab_l1 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag1(query=ag_h1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ag_h2 = ag_h1 + ab2ag_x

        ab_h2 = self.ab_gnn2(ab_h2)
        ab_l2 = self.ab_gnn2(ab_l2)
        ag_h2 = self.ag_gnn2(ag_h2)

        ag2ab_x, ag2ab_s = self.cross_attn_layer_ag2ab2(query=ab_h2, key=ag_h2, value=ag_h2, pos_embedding=ag_h2, edge_weights=ag2ab_s)
        abl2abh_x, abl2abh_s = self.cross_attn_layer_abl2h2(query=ab_h2, key=ab_l2, value=ab_l2, pos_embedding=ab_l2, edge_weights=abl2abh_s)
        ab_h3 = ab_h2 + ag2ab_x + abl2abh_x
        abh2abl_x, abh2abl_s = self.cross_attn_layer_abh2l2(query=ab_l2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=abh2abl_s)
        ab_l3 = ab_l2 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag2(query=ag_h2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=ab2ag_s)
        ag_h3 = ag_h2 + ab2ag_x

        ab_h3 = self.ab_gnn3(ab_h3)
        ab_l3 = self.ab_gnn3(ab_l3)
        ag_h3 = self.ag_gnn3(ag_h3)

        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h3, kernel_size=ag_h3.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h3, kernel_size=ab_h3.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l3, kernel_size=ab_l3.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
        # x = self.classifier(x)
        # np.save("after_pooling.npy", x.cpu().numpy()) #for normalization

        x = self.dense_concat(x)
        x=x + self.res_dense_block(x)
        if self.extra_dense:
            x = x + self.res_dense_block1(x)
            x = x + self.res_dense_block2(x)
        x = self.classifier(x)
        x = self.sigmoid(x)

        if self.output_attn_score:
            return x,[torch.sum(ag2ab_s,dim=1) * torch.sum(torch.permute(ab2ag_s,(0,1,3,2)),dim=1),torch.sum(torch.permute(abh2abl_s,(0,1,3,2)),dim=1) * torch.sum(abl2abh_s,dim=1)]
        else:
            return x


class Adaptive_Regulariz():
    def __init__(self, dropout_range=[0.,0.9],weight_range=[0.,10.],velocity=[0.02,0.001],target_deviation_ratio=0.3):
        self.dropout=0.5
        # self.weight=0.1
        self.weight = 0.0
        self.dropout_range=dropout_range
        self.weight_range=weight_range
        self.dropout_velocity=velocity[0]
        self.weight_velocity = velocity[1]
        self.thresh=target_deviation_ratio
        return

    def _update_scale(self, loss_src, loss_tgt, eps=10e-8):
        return (loss_tgt - loss_src) / (abs(loss_tgt) + eps) - self.thresh
        # return np.maximum((loss_tgt-loss_src)/abs(loss_tgt)-self.thresh,0)
        # return (abs(loss_tgt)-abs(loss_src))/abs(loss_tgt)-self.thresh

    def update_dropout(self,loss_src,loss_tgt):
        scale=self._update_scale(loss_src,loss_tgt)
        self.dropout+=self.dropout_velocity*scale
        return np.clip(self.dropout, self.dropout_range[0], self.dropout_range[1]).item()

    def update_weight(self,loss_src,loss_tgt):
        scale=self._update_scale(loss_src,loss_tgt)
        self.weight+=self.weight_velocity*scale
        return np.clip(self.weight, self.weight_range[0], self.weight_range[1]).item()


class DeepBCR_ACEXN(torch.nn.Module):

    def __init__(self, node_attr_dim=1024, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=2, ab_freeze_layer_count=None,output_attn_score=False,extra_dense=True,
                 extra_regressor=False,extra_classifier=False, print_feature=False, freeze_bert=False):
        super(DeepBCR_ACEXN, self).__init__()
        # === 0416
        # heads=16
        # expand_num = 4
        # h1_dim = 64
        # === 0417
        # heads = 16
        # expand_num = 8
        # h1_dim = 256
        #=== 0418
        heads = 64
        expand_num = 8
        h1_dim = 512

        self.output_attn_score=output_attn_score
        self.extra_dense=extra_dense
        self.extra_regressor=extra_regressor
        self.extra_classifier=extra_classifier
        self.ab_bert = BertModel.from_pretrained("train_2_193999")
        # freeze the ? layers
        if ab_freeze_layer_count is not None:
            _freeze_bert(self.ab_bert, freeze_bert=freeze_bert, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn3 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()

        self.cross_attn_layer_abh2l1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=self.heads, dropout_p=0.)
        self.cross_attn_layer_abl2h1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=self.heads, dropout_p=0.)
        self.cross_attn_layer_ag2ab1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=self.heads, dropout_p=0.)
        self.cross_attn_layer_ab2ag1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=self.heads, dropout_p=0.)
        self.cross_attn_layer_abh2l2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=self.heads, dropout_p=0.)
        self.cross_attn_layer_abl2h2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=self.heads, dropout_p=0.)
        self.cross_attn_layer_ag2ab2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=self.heads, dropout_p=0.)
        self.cross_attn_layer_ab2ag2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=self.heads, dropout_p=0.)

        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
            self.ab_gnn3 = self.ag_gnn3
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn3 = torch.nn.Sequential(*atrous_blocks)


        self.expand = Atrous_block(self.hidden_dim, expand_num*self.hidden_dim)

        self.dense_concat = torch.nn.Linear(3 * expand_num* self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        if self.extra_dense:
            self.res_dense_block1 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
            self.res_dense_block2 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.classifier = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))
        if self.extra_classifier:
            self.dense_concat_ab = torch.nn.Linear(2 * expand_num* self.hidden_dim, self.h1_dim)
            self.res_dense_block3 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
            self.res_dense_block4 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
            self.classifier_ar = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, 1))
        if self.extra_regressor:
            self.regressor = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, 1))
        self.print_feature = print_feature
        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(3 * self.hidden_dim, self.h1_dim),
        #     self.relu,
        #     torch.nn.Linear(self.h1_dim ,self.h2_dim),
        #     self.relu,
        #     torch.nn.Dropout(dropout),
        #     torch.nn.Linear(self.h2_dim, 1))
        #

    def get_variables(self,lp=1):
        self.device=next(self.dense_concat.parameters()).device
        total_regul=torch.tensor(0,dtype=torch.float,device=self.device)
        total_param_num = 1
        for param in self.parameters():
            total_regul += torch.norm(param, lp) ** 1
            total_param_num += param.numel()
        cls_x_regul,cls_y_regul=torch.tensor(0,dtype=torch.float,device=self.device),torch.tensor(0,dtype=torch.float,device=self.device)
        cls_x_param_num, cls_y_param_num = 1, 1
        if self.extra_dense:
            cls_x=[
                self.dense_concat,
                self.res_dense_block1,
                self.res_dense_block2,
                # self.classifier,
            ]
            for model in cls_x:
                # print(model)
                for param in model.parameters():
                    cls_x_regul += torch.norm(param, lp) ** 1
                    cls_x_param_num += param.numel()

        if self.extra_classifier:
            cls_y=[
                self.dense_concat_ab,
                self.res_dense_block3,
                self.res_dense_block4,
                # self.classifier_ar,
            ]
            for model in cls_y:
                for param in model.parameters():
                    cls_y_regul += torch.norm(param, lp) ** 2
                    cls_y_param_num += param.numel()
        # return [0.01*cls_x_regul/cls_x_param_num,0.01*cls_y_regul/cls_y_param_num]
        return [0.01 * total_regul / total_param_num,0.01*cls_x_regul/cls_x_param_num]

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None,attn_score=None):
        # [ag2ab_s,abl2abh_s,abh2abl_s,ab2ag_s]=attn_score
        ab_x = self.ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
        ab_l = self.ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])
        ab_x = torch.permute(ab_x, [0, 2, 1])
        ab_l = torch.permute(ab_l, [0, 2, 1])

        # antibody acnn x cross attention
        ab_h1 = self.ab_gnn1(ab_x)
        ab_l1 = self.ab_gnn1(ab_l)
        ag_h1 = self.ag_gnn1(ag_x)

        ag2ab_x,ag2ab_s = self.cross_attn_layer_ag2ab1(query=ab_h1, key=ag_h1, value=ag_h1, pos_embedding=ag_h1)
        abl2abh_x,abl2abh_s = self.cross_attn_layer_abl2h1(query=ab_h1, key=ab_l1, value=ab_l1,pos_embedding=ab_l1)
        ab_h2 = ab_h1 + ag2ab_x + abl2abh_x
        abh2abl_x,abh2abl_s = self.cross_attn_layer_abh2l1(query=ab_l1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ab_l2 = ab_l1 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag1(query=ag_h1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ag_h2 = ag_h1 + ab2ag_x

        ab_h2 = self.ab_gnn2(ab_h2)
        ab_l2 = self.ab_gnn2(ab_l2)
        ag_h2 = self.ag_gnn2(ag_h2)

        ag2ab_x, ag2ab_s = self.cross_attn_layer_ag2ab2(query=ab_h2, key=ag_h2, value=ag_h2, pos_embedding=ag_h2, edge_weights=ag2ab_s)
        abl2abh_x, abl2abh_s = self.cross_attn_layer_abl2h2(query=ab_h2, key=ab_l2, value=ab_l2, pos_embedding=ab_l2, edge_weights=abl2abh_s)
        ab_h3 = ab_h2 + ag2ab_x + abl2abh_x
        abh2abl_x, abh2abl_s = self.cross_attn_layer_abh2l2(query=ab_l2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=abh2abl_s)
        ab_l3 = ab_l2 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag2(query=ag_h2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=ab2ag_s)
        ag_h3 = ag_h2 + ab2ag_x

        ab_h3 = self.ab_gnn3(ab_h3)
        ab_l3 = self.ab_gnn3(ab_l3)
        ag_h3 = self.ag_gnn3(ag_h3)

        ab_h3 = self.expand(ab_h3)
        ab_l3 = self.expand(ab_l3)
        ag_h3 = self.expand(ag_h3)
        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h3, kernel_size=ag_h3.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h3, kernel_size=ab_h3.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l3, kernel_size=ab_l3.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        # x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
        # x = self.classifier(x)
        # np.save("after_pooling.npy", x.cpu().numpy()) #for normalization
        out = []
        x = self.dense_concat(torch.cat([x1, x2, x3], 1))
        # x=x + self.res_dense_block(x)
        if self.extra_dense:
            x = x + self.res_dense_block1(x)
            x = x + self.res_dense_block2(x)
        x0 = self.classifier(x)
        x0 = self.sigmoid(x0)
        out=out+[x0]
        if self.extra_regressor:
            # x1 = x + self.res_dense_block3(x)
            x1 = self.regressor(x)
            out=out+[x1]
        if self.extra_classifier:
            y = self.dense_concat_ab(torch.cat([x2, x3], 1))
            y = y + self.res_dense_block3(y)
            y = y + self.res_dense_block4(y)
            y = self.classifier_ar(y)
            y = self.sigmoid(y)
            out.append(y)
        if self.output_attn_score:
            attn=[torch.sum(ag2ab_s, dim=1) * torch.sum(torch.permute(ab2ag_s, (0, 1, 3, 2)), dim=1),torch.sum(torch.permute(abh2abl_s, (0, 1, 3, 2)), dim=1) * torch.sum(abl2abh_s, dim=1)]
            out.append(attn)
        if self.print_feature:
            out.append(x)
        return out
        #     if self.extra_regressor:
        #         return x0,x1,attn
        #     else:
        #         return x0,attn
        # else:
        #     if self.extra_regressor:
        #         return x0, x1
        #     else:
        #         return x0


class DeepBCR_ACEXN_protbert(torch.nn.Module):

    def __init__(self, node_attr_dim=1024, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=2, ab_freeze_layer_count=None, freeze_bert=False, output_attn_score=False,extra_dense=True,
                 extra_regressor=False,extra_classifier=False, print_feature=False):
        super(DeepBCR_ACEXN_protbert, self).__init__()
        # === 0416
        # heads=16
        # expand_num = 4
        # h1_dim = 64
        # === 0417
        # heads = 16
        # expand_num = 8
        # h1_dim = 256
        #=== 0418
        heads = 64
        expand_num = 8
        h1_dim = 512

        self.output_attn_score=output_attn_score
        self.extra_dense=extra_dense
        self.extra_regressor=extra_regressor
        self.extra_classifier=extra_classifier
        self.ab_bert = BertModel.from_pretrained("prot_bert")
        # freeze the ? layers
        if ab_freeze_layer_count is not None:
            _freeze_bert(self.ab_bert, freeze_bert=freeze_bert, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn3 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()

        self.cross_attn_layer_abh2l1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=self.heads, dropout_p=0.)
        self.cross_attn_layer_abl2h1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=self.heads, dropout_p=0.)
        self.cross_attn_layer_ag2ab1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=self.heads, dropout_p=0.)
        self.cross_attn_layer_ab2ag1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=self.heads, dropout_p=0.)
        self.cross_attn_layer_abh2l2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=self.heads, dropout_p=0.)
        self.cross_attn_layer_abl2h2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=self.heads, dropout_p=0.)
        self.cross_attn_layer_ag2ab2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=self.heads, dropout_p=0.)
        self.cross_attn_layer_ab2ag2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=self.heads, dropout_p=0.)

        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
            self.ab_gnn3 = self.ag_gnn3
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn3 = torch.nn.Sequential(*atrous_blocks)


        self.expand = Atrous_block(self.hidden_dim, expand_num*self.hidden_dim)

        self.dense_concat = torch.nn.Linear(3 * expand_num* self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        if self.extra_dense:
            self.res_dense_block1 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
            self.res_dense_block2 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.classifier = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))
        if self.extra_classifier:
            self.dense_concat_ab = torch.nn.Linear(2 * expand_num* self.hidden_dim, self.h1_dim)
            self.res_dense_block3 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
            self.res_dense_block4 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
            self.classifier_ar = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, 1))
        if self.extra_regressor:
            self.regressor = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, 1))
        self.print_feature = print_feature
        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(3 * self.hidden_dim, self.h1_dim),
        #     self.relu,
        #     torch.nn.Linear(self.h1_dim ,self.h2_dim),
        #     self.relu,
        #     torch.nn.Dropout(dropout),
        #     torch.nn.Linear(self.h2_dim, 1))
        #

    def get_variables(self,lp=1):
        self.device=next(self.dense_concat.parameters()).device
        total_regul=torch.tensor(0,dtype=torch.float,device=self.device)
        total_param_num = 1
        for param in self.parameters():
            total_regul += torch.norm(param, lp) ** 1
            total_param_num += param.numel()
        cls_x_regul,cls_y_regul=torch.tensor(0,dtype=torch.float,device=self.device),torch.tensor(0,dtype=torch.float,device=self.device)
        cls_x_param_num, cls_y_param_num = 1, 1
        if self.extra_dense:
            cls_x=[
                self.dense_concat,
                self.res_dense_block1,
                self.res_dense_block2,
                # self.classifier,
            ]
            for model in cls_x:
                # print(model)
                for param in model.parameters():
                    cls_x_regul += torch.norm(param, lp) ** 1
                    cls_x_param_num += param.numel()

        if self.extra_classifier:
            cls_y=[
                self.dense_concat_ab,
                self.res_dense_block3,
                self.res_dense_block4,
                # self.classifier_ar,
            ]
            for model in cls_y:
                for param in model.parameters():
                    cls_y_regul += torch.norm(param, lp) ** 2
                    cls_y_param_num += param.numel()
        # return [0.01*cls_x_regul/cls_x_param_num,0.01*cls_y_regul/cls_y_param_num]
        return [0.01 * total_regul / total_param_num,0.01*cls_x_regul/cls_x_param_num]

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None,attn_score=None):
        # [ag2ab_s,abl2abh_s,abh2abl_s,ab2ag_s]=attn_score
        ab_x = self.ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
        ab_l = self.ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])
        ab_x = torch.permute(ab_x, [0, 2, 1])
        ab_l = torch.permute(ab_l, [0, 2, 1])

        # antibody acnn x cross attention
        ab_h1 = self.ab_gnn1(ab_x)
        ab_l1 = self.ab_gnn1(ab_l)
        ag_h1 = self.ag_gnn1(ag_x)

        ag2ab_x,ag2ab_s = self.cross_attn_layer_ag2ab1(query=ab_h1, key=ag_h1, value=ag_h1, pos_embedding=ag_h1)
        abl2abh_x,abl2abh_s = self.cross_attn_layer_abl2h1(query=ab_h1, key=ab_l1, value=ab_l1,pos_embedding=ab_l1)
        ab_h2 = ab_h1 + ag2ab_x + abl2abh_x
        abh2abl_x,abh2abl_s = self.cross_attn_layer_abh2l1(query=ab_l1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ab_l2 = ab_l1 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag1(query=ag_h1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ag_h2 = ag_h1 + ab2ag_x

        ab_h2 = self.ab_gnn2(ab_h2)
        ab_l2 = self.ab_gnn2(ab_l2)
        ag_h2 = self.ag_gnn2(ag_h2)

        ag2ab_x, ag2ab_s = self.cross_attn_layer_ag2ab2(query=ab_h2, key=ag_h2, value=ag_h2, pos_embedding=ag_h2, edge_weights=ag2ab_s)
        abl2abh_x, abl2abh_s = self.cross_attn_layer_abl2h2(query=ab_h2, key=ab_l2, value=ab_l2, pos_embedding=ab_l2, edge_weights=abl2abh_s)
        ab_h3 = ab_h2 + ag2ab_x + abl2abh_x
        abh2abl_x, abh2abl_s = self.cross_attn_layer_abh2l2(query=ab_l2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=abh2abl_s)
        ab_l3 = ab_l2 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag2(query=ag_h2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=ab2ag_s)
        ag_h3 = ag_h2 + ab2ag_x

        ab_h3 = self.ab_gnn3(ab_h3)
        ab_l3 = self.ab_gnn3(ab_l3)
        ag_h3 = self.ag_gnn3(ag_h3)

        ab_h3 = self.expand(ab_h3)
        ab_l3 = self.expand(ab_l3)
        ag_h3 = self.expand(ag_h3)
        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h3, kernel_size=ag_h3.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h3, kernel_size=ab_h3.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l3, kernel_size=ab_l3.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        # x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
        # x = self.classifier(x)
        # np.save("after_pooling.npy", x.cpu().numpy()) #for normalization
        out = []
        x = self.dense_concat(torch.cat([x1, x2, x3], 1))
        # x=x + self.res_dense_block(x)
        if self.extra_dense:
            x = x + self.res_dense_block1(x)
            x = x + self.res_dense_block2(x)
        x0 = self.classifier(x)
        x0 = self.sigmoid(x0)
        out=out+[x0]
        if self.extra_regressor:
            # x1 = x + self.res_dense_block3(x)
            x1 = self.regressor(x)
            out=out+[x1]
        if self.extra_classifier:
            y = self.dense_concat_ab(torch.cat([x2, x3], 1))
            y = y + self.res_dense_block3(y)
            y = y + self.res_dense_block4(y)
            y = self.classifier_ar(y)
            y = self.sigmoid(y)
            out.append(y)
        if self.output_attn_score:
            attn=[torch.sum(ag2ab_s, dim=1) * torch.sum(torch.permute(ab2ag_s, (0, 1, 3, 2)), dim=1),torch.sum(torch.permute(abh2abl_s, (0, 1, 3, 2)), dim=1) * torch.sum(abl2abh_s, dim=1)]
            out.append(attn)
        if self.print_feature:
            out.append(x)
        return out
        #     if self.extra_regressor:
        #         return x0,x1,attn
        #     else:
        #         return x0,attn
        # else:
        #     if self.extra_regressor:
        #         return x0, x1
        #     else:
        #         return x0

class XBCR_ACEXN_ED_neu(torch.nn.Module):

    def __init__(self, node_attr_dim=1024, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=2, ab_freeze_layer_count=None,output_attn_score=False,extra_dense=True):
        super(XBCR_ACEXN_ED_neu, self).__init__()
        self.output_attn_score=output_attn_score
        self.extra_dense=extra_dense
        self.ab_bert = BertModel.from_pretrained("train_2_193999")
        # freeze the ? layers
        if ab_freeze_layer_count is not None:
            _freeze_bert(self.ab_bert, freeze_bert=False, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn3 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()

        self.cross_attn_layer_abh2l1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abl2h1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag2ab1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab2ag1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abh2l2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abl2h2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag2ab2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab2ag2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)

        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
            self.ab_gnn3 = self.ag_gnn3
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn3 = torch.nn.Sequential(*atrous_blocks)

        self.dense_concat = torch.nn.Linear(3 * self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        if self.extra_dense:
            self.res_dense_block1 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
            self.res_dense_block2 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
        # self.classifier = torch.nn.Sequential(
        #     self.relu,
        #     torch.nn.Linear(self.h1_dim, 1))
        self.classifier_neu = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(3 * self.hidden_dim, self.h1_dim),
        #     self.relu,
        #     torch.nn.Linear(self.h1_dim ,self.h2_dim),
        #     self.relu,
        #     torch.nn.Dropout(dropout),
        #     torch.nn.Linear(self.h2_dim, 1))
        #

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None,attn_score=None):
        # [ag2ab_s,abl2abh_s,abh2abl_s,ab2ag_s]=attn_score
        ab_x = self.ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
        ab_l = self.ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])
        ab_x = torch.permute(ab_x, [0, 2, 1])
        ab_l = torch.permute(ab_l, [0, 2, 1])

        # antibody acnn x cross attention
        ab_h1 = self.ab_gnn1(ab_x)
        ab_l1 = self.ab_gnn1(ab_l)
        ag_h1 = self.ag_gnn1(ag_x)

        ag2ab_x,ag2ab_s = self.cross_attn_layer_ag2ab1(query=ab_h1, key=ag_h1, value=ag_h1, pos_embedding=ag_h1)
        abl2abh_x,abl2abh_s = self.cross_attn_layer_abl2h1(query=ab_h1, key=ab_l1, value=ab_l1,pos_embedding=ab_l1)
        ab_h2 = ab_h1 + ag2ab_x + abl2abh_x
        abh2abl_x,abh2abl_s = self.cross_attn_layer_abh2l1(query=ab_l1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ab_l2 = ab_l1 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag1(query=ag_h1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ag_h2 = ag_h1 + ab2ag_x

        ab_h2 = self.ab_gnn2(ab_h2)
        ab_l2 = self.ab_gnn2(ab_l2)
        ag_h2 = self.ag_gnn2(ag_h2)

        ag2ab_x, ag2ab_s = self.cross_attn_layer_ag2ab2(query=ab_h2, key=ag_h2, value=ag_h2, pos_embedding=ag_h2, edge_weights=ag2ab_s)
        abl2abh_x, abl2abh_s = self.cross_attn_layer_abl2h2(query=ab_h2, key=ab_l2, value=ab_l2, pos_embedding=ab_l2, edge_weights=abl2abh_s)
        ab_h3 = ab_h2 + ag2ab_x + abl2abh_x
        abh2abl_x, abh2abl_s = self.cross_attn_layer_abh2l2(query=ab_l2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=abh2abl_s)
        ab_l3 = ab_l2 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag2(query=ag_h2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=ab2ag_s)
        ag_h3 = ag_h2 + ab2ag_x

        ab_h3 = self.ab_gnn3(ab_h3)
        ab_l3 = self.ab_gnn3(ab_l3)
        ag_h3 = self.ag_gnn3(ag_h3)

        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h3, kernel_size=ag_h3.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h3, kernel_size=ab_h3.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l3, kernel_size=ab_l3.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
        # x = self.classifier(x)

        x = self.dense_concat(x)
        x=x + self.res_dense_block(x)
        if self.extra_dense:
            x = x + self.res_dense_block1(x)
            x = x + self.res_dense_block2(x)
        # x = self.classifier(x)
        # x = self.sigmoid(x)
        x = self.classifier_neu(x)
        x = self.sigmoid(x)

        if self.output_attn_score:
            return x,[torch.sum(ag2ab_s,dim=1) * torch.sum(torch.permute(ab2ag_s,(0,1,3,2)),dim=1),torch.sum(torch.permute(abh2abl_s,(0,1,3,2)),dim=1) * torch.sum(abl2abh_s,dim=1)]
        else:
            return x

class XBCR_ACEXN_ED_neu_newbert(torch.nn.Module):

    def __init__(self, node_attr_dim=1024, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=2, ab_freeze_layer_count=None,output_attn_score=False,extra_dense=True):
        super(XBCR_ACEXN_ED_neu_newbert, self).__init__()
        self.output_attn_score=output_attn_score
        self.extra_dense=extra_dense
        self.ab_bert_new = BertModel.from_pretrained("train_2_193999")
        # freeze the ? layers
        if ab_freeze_layer_count is not None:
            _freeze_bert(self.ab_bert_new, freeze_bert=False, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn3 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()

        self.cross_attn_layer_abh2l1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abl2h1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag2ab1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab2ag1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abh2l2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abl2h2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag2ab2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab2ag2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)

        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
            self.ab_gnn3 = self.ag_gnn3
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn3 = torch.nn.Sequential(*atrous_blocks)

        self.dense_concat = torch.nn.Linear(3 * self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        if self.extra_dense:
            self.res_dense_block1 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
            self.res_dense_block2 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
        # self.classifier = torch.nn.Sequential(
        #     self.relu,
        #     torch.nn.Linear(self.h1_dim, 1))
        self.classifier_neu = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(3 * self.hidden_dim, self.h1_dim),
        #     self.relu,
        #     torch.nn.Linear(self.h1_dim ,self.h2_dim),
        #     self.relu,
        #     torch.nn.Dropout(dropout),
        #     torch.nn.Linear(self.h2_dim, 1))
        #

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None,attn_score=None):
        # [ag2ab_s,abl2abh_s,abh2abl_s,ab2ag_s]=attn_score
        ab_x = self.ab_bert_new(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
        ab_l = self.ab_bert_new(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])
        ab_x = torch.permute(ab_x, [0, 2, 1])
        ab_l = torch.permute(ab_l, [0, 2, 1])

        # antibody acnn x cross attention
        ab_h1 = self.ab_gnn1(ab_x)
        ab_l1 = self.ab_gnn1(ab_l)
        ag_h1 = self.ag_gnn1(ag_x)

        ag2ab_x,ag2ab_s = self.cross_attn_layer_ag2ab1(query=ab_h1, key=ag_h1, value=ag_h1, pos_embedding=ag_h1)
        abl2abh_x,abl2abh_s = self.cross_attn_layer_abl2h1(query=ab_h1, key=ab_l1, value=ab_l1,pos_embedding=ab_l1)
        ab_h2 = ab_h1 + ag2ab_x + abl2abh_x
        abh2abl_x,abh2abl_s = self.cross_attn_layer_abh2l1(query=ab_l1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ab_l2 = ab_l1 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag1(query=ag_h1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ag_h2 = ag_h1 + ab2ag_x

        ab_h2 = self.ab_gnn2(ab_h2)
        ab_l2 = self.ab_gnn2(ab_l2)
        ag_h2 = self.ag_gnn2(ag_h2)

        ag2ab_x, ag2ab_s = self.cross_attn_layer_ag2ab2(query=ab_h2, key=ag_h2, value=ag_h2, pos_embedding=ag_h2, edge_weights=ag2ab_s)
        abl2abh_x, abl2abh_s = self.cross_attn_layer_abl2h2(query=ab_h2, key=ab_l2, value=ab_l2, pos_embedding=ab_l2, edge_weights=abl2abh_s)
        ab_h3 = ab_h2 + ag2ab_x + abl2abh_x
        abh2abl_x, abh2abl_s = self.cross_attn_layer_abh2l2(query=ab_l2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=abh2abl_s)
        ab_l3 = ab_l2 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag2(query=ag_h2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=ab2ag_s)
        ag_h3 = ag_h2 + ab2ag_x

        ab_h3 = self.ab_gnn3(ab_h3)
        ab_l3 = self.ab_gnn3(ab_l3)
        ag_h3 = self.ag_gnn3(ag_h3)

        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h3, kernel_size=ag_h3.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h3, kernel_size=ab_h3.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l3, kernel_size=ab_l3.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
        # x = self.classifier(x)
        # np.save("after_pooling_abh.npy", x2.cpu().numpy())
        # np.save("after_pooling_abl.npy", x3.cpu().numpy())
        # np.save("after_poolingcat1.npy", x.cpu().numpy())  # for normalization

        x = self.dense_concat(x)
        # np.save("after_poolingcat2.npy", x.cpu().numpy())
        x=x + self.res_dense_block(x)
        # np.save("after_poolingcat3.npy", x.cpu().numpy())
        if self.extra_dense:
            x = x + self.res_dense_block1(x)
            x = x + self.res_dense_block2(x)
        # x = self.classifier(x)
        # x = self.sigmoid(x)
        # np.save("after_poolingcat4.npy", x.cpu().numpy())
        x = self.classifier_neu(x)
        # np.save("after_poolingcat5.npy", x.cpu().numpy())
        x = self.sigmoid(x)

        if self.output_attn_score:
            return x,[torch.sum(ag2ab_s,dim=1) * torch.sum(torch.permute(ab2ag_s,(0,1,3,2)),dim=1),torch.sum(torch.permute(abh2abl_s,(0,1,3,2)),dim=1) * torch.sum(abl2abh_s,dim=1)]
        else:
            return x

class XBCR_ACEXN_ED_neu_newbert_addfeature(torch.nn.Module):

    def __init__(self, node_attr_dim=1024, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=2, ab_freeze_layer_count=None,output_attn_score=False,extra_dense=True):
        super(XBCR_ACEXN_ED_neu_newbert_addfeature, self).__init__()
        self.output_attn_score=output_attn_score
        self.extra_dense=extra_dense
        self.ab_bert_new = BertModel.from_pretrained("train_2_193999")
        # freeze the ? layers
        if ab_freeze_layer_count is not None:
            _freeze_bert(self.ab_bert_new, freeze_bert=False, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn3 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()

        self.cross_attn_layer_abh2l1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abl2h1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag2ab1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab2ag1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abh2l2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abl2h2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag2ab2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab2ag2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)

        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
            self.ab_gnn3 = self.ag_gnn3
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn3 = torch.nn.Sequential(*atrous_blocks)

        self.dense_concat = torch.nn.Linear(3 * self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        if self.extra_dense:
            self.res_dense_block1 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
            self.res_dense_block2 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
        # self.classifier = torch.nn.Sequential(
        #     self.relu,
        #     torch.nn.Linear(self.h1_dim, 1))

        self.classifier_neu = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(128, 1))

        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(3 * self.hidden_dim, self.h1_dim),
        #     self.relu,
        #     torch.nn.Linear(self.h1_dim ,self.h2_dim),
        #     self.relu,
        #     torch.nn.Dropout(dropout),
        #     torch.nn.Linear(self.h2_dim, 1))
        #

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None,attn_score=None,charge=None,hydro=None):
        # [ag2ab_s,abl2abh_s,abh2abl_s,ab2ag_s]=attn_score
        ab_x = self.ab_bert_new(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
        ab_l = self.ab_bert_new(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])
        ab_x = torch.permute(ab_x, [0, 2, 1])
        ab_l = torch.permute(ab_l, [0, 2, 1])

        # antibody acnn x cross attention
        ab_h1 = self.ab_gnn1(ab_x)
        ab_l1 = self.ab_gnn1(ab_l)
        ag_h1 = self.ag_gnn1(ag_x)

        ag2ab_x,ag2ab_s = self.cross_attn_layer_ag2ab1(query=ab_h1, key=ag_h1, value=ag_h1, pos_embedding=ag_h1)
        abl2abh_x,abl2abh_s = self.cross_attn_layer_abl2h1(query=ab_h1, key=ab_l1, value=ab_l1,pos_embedding=ab_l1)
        ab_h2 = ab_h1 + ag2ab_x + abl2abh_x
        abh2abl_x,abh2abl_s = self.cross_attn_layer_abh2l1(query=ab_l1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ab_l2 = ab_l1 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag1(query=ag_h1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ag_h2 = ag_h1 + ab2ag_x

        ab_h2 = self.ab_gnn2(ab_h2)
        ab_l2 = self.ab_gnn2(ab_l2)
        ag_h2 = self.ag_gnn2(ag_h2)

        ag2ab_x, ag2ab_s = self.cross_attn_layer_ag2ab2(query=ab_h2, key=ag_h2, value=ag_h2, pos_embedding=ag_h2, edge_weights=ag2ab_s)
        abl2abh_x, abl2abh_s = self.cross_attn_layer_abl2h2(query=ab_h2, key=ab_l2, value=ab_l2, pos_embedding=ab_l2, edge_weights=abl2abh_s)
        ab_h3 = ab_h2 + ag2ab_x + abl2abh_x
        abh2abl_x, abh2abl_s = self.cross_attn_layer_abh2l2(query=ab_l2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=abh2abl_s)
        ab_l3 = ab_l2 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag2(query=ag_h2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=ab2ag_s)
        ag_h3 = ag_h2 + ab2ag_x

        ab_h3 = self.ab_gnn3(ab_h3)
        ab_l3 = self.ab_gnn3(ab_l3)
        ag_h3 = self.ag_gnn3(ag_h3)

        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h3, kernel_size=ag_h3.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h3, kernel_size=ab_h3.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l3, kernel_size=ab_l3.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
        # x = self.classifier(x)
        # np.save("after_pooling_abh.npy", x2.cpu().numpy())
        # np.save("after_pooling_abl.npy", x3.cpu().numpy())
        # np.save("after_poolingcat1.npy", x.cpu().numpy())  # for normalization

        x = self.dense_concat(x)
        # np.save("after_poolingcat2.npy", x.cpu().numpy())
        x=x + self.res_dense_block(x)
        # np.save("after_poolingcat3.npy", x.cpu().numpy())
        if self.extra_dense:
            x = x + self.res_dense_block1(x)
            x = x + self.res_dense_block2(x)
        # x = self.classifier(x)
        # x = self.sigmoid(x)
        # np.save("after_poolingcat4.npy", x.cpu().numpy())
        # print(charge.unsqueeze(-1).shape)
        # print(hydro.unsqueeze(-1).shape)
        x = torch.cat([torch.multiply(x,charge.unsqueeze(-1)), torch.multiply(x,hydro.unsqueeze(-1))],1)
        # print(x.shape)
        x = self.classifier_neu(x)
        # np.save("after_poolingcat5.npy", x.cpu().numpy())
        x = self.sigmoid(x)

        if self.output_attn_score:
            return x,[torch.sum(ag2ab_s,dim=1) * torch.sum(torch.permute(ab2ag_s,(0,1,3,2)),dim=1),torch.sum(torch.permute(abh2abl_s,(0,1,3,2)),dim=1) * torch.sum(abl2abh_s,dim=1)]
        else:
            return x

class XBCR_ACEXN_ED_neu_newbert_addfeature2(torch.nn.Module):

    def __init__(self, node_attr_dim=1024, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=2, ab_freeze_layer_count=None,output_attn_score=False,extra_dense=True):
        super(XBCR_ACEXN_ED_neu_newbert_addfeature2, self).__init__()
        self.output_attn_score=output_attn_score
        self.extra_dense=extra_dense
        self.ab_bert_new = BertModel.from_pretrained("train_2_193999")
        # freeze the ? layers
        if ab_freeze_layer_count is not None:
            _freeze_bert(self.ab_bert_new, freeze_bert=False, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn3 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()

        self.cross_attn_layer_abh2l1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abl2h1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag2ab1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab2ag1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abh2l2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abl2h2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag2ab2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab2ag2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)

        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
            self.ab_gnn3 = self.ag_gnn3
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn3 = torch.nn.Sequential(*atrous_blocks)

        self.dense_concat = torch.nn.Linear(3 * self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        if self.extra_dense:
            self.res_dense_block1 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
            self.res_dense_block2 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
        # self.classifier = torch.nn.Sequential(
        #     self.relu,
        #     torch.nn.Linear(self.h1_dim, 1))

        self.classifier_neu = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(128, 16),
            self.relu,
            torch.nn.Linear(16, 1))

        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(3 * self.hidden_dim, self.h1_dim),
        #     self.relu,
        #     torch.nn.Linear(self.h1_dim ,self.h2_dim),
        #     self.relu,
        #     torch.nn.Dropout(dropout),
        #     torch.nn.Linear(self.h2_dim, 1))
        #

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None,attn_score=None,charge=None,hydro=None):
        # [ag2ab_s,abl2abh_s,abh2abl_s,ab2ag_s]=attn_score
        ab_x = self.ab_bert_new(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
        ab_l = self.ab_bert_new(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])
        ab_x = torch.permute(ab_x, [0, 2, 1])
        ab_l = torch.permute(ab_l, [0, 2, 1])

        # antibody acnn x cross attention
        ab_h1 = self.ab_gnn1(ab_x)
        ab_l1 = self.ab_gnn1(ab_l)
        ag_h1 = self.ag_gnn1(ag_x)

        ag2ab_x,ag2ab_s = self.cross_attn_layer_ag2ab1(query=ab_h1, key=ag_h1, value=ag_h1, pos_embedding=ag_h1)
        abl2abh_x,abl2abh_s = self.cross_attn_layer_abl2h1(query=ab_h1, key=ab_l1, value=ab_l1,pos_embedding=ab_l1)
        ab_h2 = ab_h1 + ag2ab_x + abl2abh_x
        abh2abl_x,abh2abl_s = self.cross_attn_layer_abh2l1(query=ab_l1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ab_l2 = ab_l1 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag1(query=ag_h1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ag_h2 = ag_h1 + ab2ag_x

        ab_h2 = self.ab_gnn2(ab_h2)
        ab_l2 = self.ab_gnn2(ab_l2)
        ag_h2 = self.ag_gnn2(ag_h2)

        ag2ab_x, ag2ab_s = self.cross_attn_layer_ag2ab2(query=ab_h2, key=ag_h2, value=ag_h2, pos_embedding=ag_h2, edge_weights=ag2ab_s)
        abl2abh_x, abl2abh_s = self.cross_attn_layer_abl2h2(query=ab_h2, key=ab_l2, value=ab_l2, pos_embedding=ab_l2, edge_weights=abl2abh_s)
        ab_h3 = ab_h2 + ag2ab_x + abl2abh_x
        abh2abl_x, abh2abl_s = self.cross_attn_layer_abh2l2(query=ab_l2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=abh2abl_s)
        ab_l3 = ab_l2 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag2(query=ag_h2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=ab2ag_s)
        ag_h3 = ag_h2 + ab2ag_x

        ab_h3 = self.ab_gnn3(ab_h3)
        ab_l3 = self.ab_gnn3(ab_l3)
        ag_h3 = self.ag_gnn3(ag_h3)

        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h3, kernel_size=ag_h3.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h3, kernel_size=ab_h3.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l3, kernel_size=ab_l3.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
        # x = self.classifier(x)
        # np.save("after_pooling_abh.npy", x2.cpu().numpy())
        # np.save("after_pooling_abl.npy", x3.cpu().numpy())
        # np.save("after_poolingcat1.npy", x.cpu().numpy())  # for normalization

        x = self.dense_concat(x)
        # np.save("after_poolingcat2.npy", x.cpu().numpy())
        x=x + self.res_dense_block(x)
        # np.save("after_poolingcat3.npy", x.cpu().numpy())
        if self.extra_dense:
            x = x + self.res_dense_block1(x)
            x = x + self.res_dense_block2(x)
        # x = self.classifier(x)
        # x = self.sigmoid(x)
        # np.save("after_poolingcat4.npy", x.cpu().numpy())
        # print(charge.unsqueeze(-1).shape)
        # print(hydro.unsqueeze(-1).shape)
        x = torch.cat([torch.multiply(x,charge.unsqueeze(-1)), torch.multiply(x,hydro.unsqueeze(-1))],1)
        # print(x.shape)
        x = self.classifier_neu(x)
        # np.save("after_poolingcat5.npy", x.cpu().numpy())
        x = self.sigmoid(x)

        if self.output_attn_score:
            return x,[torch.sum(ag2ab_s,dim=1) * torch.sum(torch.permute(ab2ag_s,(0,1,3,2)),dim=1),torch.sum(torch.permute(abh2abl_s,(0,1,3,2)),dim=1) * torch.sum(abl2abh_s,dim=1)]
        else:
            return x

class XBCR_ACEXN_ED_neu_newbert_addfeature_0406(torch.nn.Module):

    def __init__(self, node_attr_dim=1024, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=2, ab_freeze_layer_count=None,output_attn_score=False,extra_dense=True,num_feature=2):
        super(XBCR_ACEXN_ED_neu_newbert_addfeature_0406, self).__init__()
        self.num_feature = num_feature
        self.output_attn_score=output_attn_score
        self.extra_dense=extra_dense
        self.ab_bert_new = BertModel.from_pretrained("train_2_193999")
        # freeze the ? layers
        if ab_freeze_layer_count is not None:
            _freeze_bert(self.ab_bert_new, freeze_bert=False, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn3 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()

        self.cross_attn_layer_abh2l1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abl2h1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag2ab1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab2ag1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abh2l2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abl2h2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag2ab2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab2ag2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)

        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
            self.ab_gnn3 = self.ag_gnn3
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn3 = torch.nn.Sequential(*atrous_blocks)

        self.dense_concat = torch.nn.Linear(3 * self.hidden_dim, self.h1_dim)
        self.fcn = torch.nn.Linear(self.num_feature, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        if self.extra_dense:
            self.res_dense_block1 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
            self.res_dense_block2 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
        # self.classifier = torch.nn.Sequential(
        #     self.relu,
        #     torch.nn.Linear(self.h1_dim, 1))

        self.classifier_neu = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(3 * self.hidden_dim, self.h1_dim),
        #     self.relu,
        #     torch.nn.Linear(self.h1_dim ,self.h2_dim),
        #     self.relu,
        #     torch.nn.Dropout(dropout),
        #     torch.nn.Linear(self.h2_dim, 1))
        #

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None,attn_score=None,charge=None,hydro=None):
        # [ag2ab_s,abl2abh_s,abh2abl_s,ab2ag_s]=attn_score
        ab_x = self.ab_bert_new(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
        ab_l = self.ab_bert_new(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])
        ab_x = torch.permute(ab_x, [0, 2, 1])
        ab_l = torch.permute(ab_l, [0, 2, 1])

        # antibody acnn x cross attention
        ab_h1 = self.ab_gnn1(ab_x)
        ab_l1 = self.ab_gnn1(ab_l)
        ag_h1 = self.ag_gnn1(ag_x)

        ag2ab_x,ag2ab_s = self.cross_attn_layer_ag2ab1(query=ab_h1, key=ag_h1, value=ag_h1, pos_embedding=ag_h1)
        abl2abh_x,abl2abh_s = self.cross_attn_layer_abl2h1(query=ab_h1, key=ab_l1, value=ab_l1,pos_embedding=ab_l1)
        ab_h2 = ab_h1 + ag2ab_x + abl2abh_x
        abh2abl_x,abh2abl_s = self.cross_attn_layer_abh2l1(query=ab_l1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ab_l2 = ab_l1 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag1(query=ag_h1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ag_h2 = ag_h1 + ab2ag_x

        ab_h2 = self.ab_gnn2(ab_h2)
        ab_l2 = self.ab_gnn2(ab_l2)
        ag_h2 = self.ag_gnn2(ag_h2)

        ag2ab_x, ag2ab_s = self.cross_attn_layer_ag2ab2(query=ab_h2, key=ag_h2, value=ag_h2, pos_embedding=ag_h2, edge_weights=ag2ab_s)
        abl2abh_x, abl2abh_s = self.cross_attn_layer_abl2h2(query=ab_h2, key=ab_l2, value=ab_l2, pos_embedding=ab_l2, edge_weights=abl2abh_s)
        ab_h3 = ab_h2 + ag2ab_x + abl2abh_x
        abh2abl_x, abh2abl_s = self.cross_attn_layer_abh2l2(query=ab_l2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=abh2abl_s)
        ab_l3 = ab_l2 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag2(query=ag_h2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=ab2ag_s)
        ag_h3 = ag_h2 + ab2ag_x

        ab_h3 = self.ab_gnn3(ab_h3)
        ab_l3 = self.ab_gnn3(ab_l3)
        ag_h3 = self.ag_gnn3(ag_h3)

        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h3, kernel_size=ag_h3.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h3, kernel_size=ab_h3.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l3, kernel_size=ab_l3.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
        # x = self.classifier(x)
        # np.save("after_pooling_abh.npy", x2.cpu().numpy())
        # np.save("after_pooling_abl.npy", x3.cpu().numpy())
        # np.save("after_poolingcat1.npy", x.cpu().numpy())  # for normalization

        # x = self.dense_concat(x)
        # add features
        # print([charge, hydro])
        # print(torch.stack([charge, hydro], 1).shape)
        x = self.dense_concat(x) + self.fcn(torch.stack([charge, hydro], 1))
        # np.save("after_poolingcat2.npy", x.cpu().numpy())
        x=x + self.res_dense_block(x)
        # np.save("after_poolingcat3.npy", x.cpu().numpy())

        if self.extra_dense:
            x = x + self.res_dense_block1(x)
            x = x + self.res_dense_block2(x)
        # x = self.classifier(x)
        # x = self.sigmoid(x)
        # np.save("after_poolingcat4.npy", x.cpu().numpy())
        # print(charge.unsqueeze(-1).shape)
        # print(hydro.unsqueeze(-1).shape)
        # print(x.shape)
        x = self.classifier_neu(x)
        # np.save("after_poolingcat5.npy", x.cpu().numpy())
        x = self.sigmoid(x)

        if self.output_attn_score:
            return x,[torch.sum(ag2ab_s,dim=1) * torch.sum(torch.permute(ab2ag_s,(0,1,3,2)),dim=1),torch.sum(torch.permute(abh2abl_s,(0,1,3,2)),dim=1) * torch.sum(abl2abh_s,dim=1)]
        else:
            return x


class XBCR_ACEXN_woBERT(torch.nn.Module):

    def __init__(self, node_attr_dim=20, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=True, dropout=0.,
                 heads=4, block_num=3, ab_freeze_layer_count=None,output_attn_score=False,extra_dense=False):
        super(XBCR_ACEXN_woBERT, self).__init__()
        self.output_attn_score=output_attn_score
        self.extra_dense=extra_dense
        # self.ab_bert = BertModel.from_pretrained("train_2_193999")
        # freeze the ? layers
        # if ab_freeze_layer_count is not None:
        #     _freeze_bert(self.ab_bert, freeze_bert=False, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        # self.relu = torch.nn.LeakyReLU()
        self.relu = torch.nn.ReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)

        # atrous_blocks = [Atrous_block(self.node_attr_dim, self.hidden_dim) for _ in range(block_num)] # 2.16
        atrous_blocks = [Atrous_block(self.node_attr_dim, self.hidden_dim)] + [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num-1)]
        self.ag_gnn1 = torch.nn.Sequential(*atrous_blocks)
        # self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn3 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()

        self.cross_attn_layer_abh2l1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abl2h1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag2ab1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab2ag1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abh2l2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abl2h2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag2ab2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab2ag2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)

        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
            self.ab_gnn3 = self.ag_gnn3
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn3 = torch.nn.Sequential(*atrous_blocks)

        self.dense_concat = torch.nn.Linear(3 * self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            # torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        if self.extra_dense:
            self.res_dense_block1 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
            self.res_dense_block2 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.classifier = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(3 * self.hidden_dim, self.h1_dim),
        #     self.relu,
        #     torch.nn.Linear(self.h1_dim ,self.h2_dim),
        #     self.relu,
        #     torch.nn.Dropout(dropout),
        #     torch.nn.Linear(self.h2_dim, 1))
        #

    def get_variables(self,lp=1):
        self.device=next(self.dense_concat.parameters()).device
        total_regul=torch.tensor(0,dtype=torch.float,device=self.device)
        total_param_num = 1
        for param in self.parameters():
            total_regul += torch.norm(param, lp) ** 1
            total_param_num += param.numel()
        cls_x_regul,cls_y_regul=torch.tensor(0,dtype=torch.float,device=self.device),torch.tensor(0,dtype=torch.float,device=self.device)
        cls_x_param_num, cls_y_param_num = 1, 1
        if self.extra_dense:
            cls_x=[
                self.dense_concat,
                self.res_dense_block1,
                self.res_dense_block2,
                # self.classifier,
            ]
            for model in cls_x:
                # print(model)
                for param in model.parameters():
                    cls_x_regul += torch.norm(param, lp) ** 1
                    cls_x_param_num += param.numel()

        # if self.extra_classifier:
        #     cls_y=[
        #         self.dense_concat_ab,
        #         self.res_dense_block3,
        #         self.res_dense_block4,
        #         # self.classifier_ar,
        #     ]
        #     for model in cls_y:
        #         for param in model.parameters():
        #             cls_y_regul += torch.norm(param, lp) ** 2
        #             cls_y_param_num += param.numel()
        # return [0.01*cls_x_regul/cls_x_param_num,0.01*cls_y_regul/cls_y_param_num]
        return [0.01 * total_regul / total_param_num,0.01*cls_x_regul/cls_x_param_num]

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None,attn_score=None):
        # [ag2ab_s,abl2abh_s,abh2abl_s,ab2ag_s]=attn_score
        # ab_x = self.ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
        # ab_l = self.ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])

        ab_x = torch.permute(ab_x, [0, 2, 1])
        # print(ab_x.shape)
        ab_l = torch.permute(ab_l, [0, 2, 1])

        # antibody acnn x cross attention
        ab_h1 = self.ab_gnn1(ab_x) # torch.Size([24, 200, 20])
        # print(ab_h1.shape)
        ab_l1 = self.ab_gnn1(ab_l)
        ag_h1 = self.ag_gnn1(ag_x)

        ag2ab_x,ag2ab_s = self.cross_attn_layer_ag2ab1(query=ab_h1, key=ag_h1, value=ag_h1, pos_embedding=ag_h1)
        abl2abh_x,abl2abh_s = self.cross_attn_layer_abl2h1(query=ab_h1, key=ab_l1, value=ab_l1,pos_embedding=ab_l1)
        ab_h2 = ab_h1 + ag2ab_x + abl2abh_x
        abh2abl_x,abh2abl_s = self.cross_attn_layer_abh2l1(query=ab_l1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ab_l2 = ab_l1 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag1(query=ag_h1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ag_h2 = ag_h1 + ab2ag_x

        ab_h2 = self.ab_gnn2(ab_h2)
        ab_l2 = self.ab_gnn2(ab_l2)
        ag_h2 = self.ag_gnn2(ag_h2)

        ag2ab_x, ag2ab_s = self.cross_attn_layer_ag2ab2(query=ab_h2, key=ag_h2, value=ag_h2, pos_embedding=ag_h2, edge_weights=ag2ab_s)
        abl2abh_x, abl2abh_s = self.cross_attn_layer_abl2h2(query=ab_h2, key=ab_l2, value=ab_l2, pos_embedding=ab_l2, edge_weights=abl2abh_s)
        ab_h3 = ab_h2 + ag2ab_x + abl2abh_x
        abh2abl_x, abh2abl_s = self.cross_attn_layer_abh2l2(query=ab_l2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=abh2abl_s)
        ab_l3 = ab_l2 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag2(query=ag_h2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=ab2ag_s)
        ag_h3 = ag_h2 + ab2ag_x

        ab_h3 = self.ab_gnn3(ab_h3)
        ab_l3 = self.ab_gnn3(ab_l3)
        ag_h3 = self.ag_gnn3(ag_h3)

        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h3, kernel_size=ag_h3.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h3, kernel_size=ab_h3.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l3, kernel_size=ab_l3.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, x3], 1)  # batch_size * 124

        # x = self.classifier(x)

        x = self.dense_concat(x)
        x=x + self.res_dense_block(x)
        if self.extra_dense:
            x = x + self.res_dense_block1(x)
            x = x + self.res_dense_block2(x)
        x = self.classifier(x)
        x = self.sigmoid(x)
        if self.output_attn_score:
            return x,[torch.sum(ag2ab_s,dim=1) * torch.sum(torch.permute(ab2ag_s,(0,1,3,2)),dim=1),torch.sum(torch.permute(abh2abl_s,(0,1,3,2)),dim=1) * torch.sum(abl2abh_s,dim=1)]
        else:
            return [x]

class big_XBCR_ACEXN_woBERT(torch.nn.Module):

    def __init__(self, node_attr_dim=20, hidden_dim=256, h1_dim=512, h2_dim=16, share_weight=True, dropout=0.,
                 heads=32, block_num=3, expand_num=4, ab_freeze_layer_count=None,output_attn_score=False,extra_dense=False):
        super(big_XBCR_ACEXN_woBERT, self).__init__()
        self.output_attn_score=output_attn_score
        self.extra_dense=extra_dense
        self.expand_num = expand_num
        # self.ab_bert = BertModel.from_pretrained("train_2_193999")
        # freeze the ? layers
        # if ab_freeze_layer_count is not None:
        #     _freeze_bert(self.ab_bert, freeze_bert=False, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        # self.relu = torch.nn.LeakyReLU()
        self.relu = torch.nn.ReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)

        # atrous_blocks = [Atrous_block(self.node_attr_dim, self.hidden_dim) for _ in range(block_num)] # 2.16
        atrous_blocks = [Atrous_block(self.node_attr_dim, self.hidden_dim)] + [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num-1)]
        self.ag_gnn1 = torch.nn.Sequential(*atrous_blocks)
        # self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn3 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()

        self.cross_attn_layer_abh2l1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abl2h1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag2ab1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab2ag1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abh2l2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abl2h2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag2ab2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab2ag2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)

        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
            self.ab_gnn3 = self.ag_gnn3
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn3 = torch.nn.Sequential(*atrous_blocks)

        self.expand = Atrous_block(self.hidden_dim, self.expand_num * self.hidden_dim)

        self.dense_concat = torch.nn.Linear(3 * self.expand_num * self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            # torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        if self.extra_dense:
            self.res_dense_block1 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
            self.res_dense_block2 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.classifier = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(3 * self.hidden_dim, self.h1_dim),
        #     self.relu,
        #     torch.nn.Linear(self.h1_dim ,self.h2_dim),
        #     self.relu,
        #     torch.nn.Dropout(dropout),
        #     torch.nn.Linear(self.h2_dim, 1))
        #

    def get_variables(self,lp=1):
        self.device=next(self.dense_concat.parameters()).device
        total_regul=torch.tensor(0,dtype=torch.float,device=self.device)
        total_param_num = 1
        for param in self.parameters():
            total_regul += torch.norm(param, lp) ** 1
            total_param_num += param.numel()
        cls_x_regul,cls_y_regul=torch.tensor(0,dtype=torch.float,device=self.device),torch.tensor(0,dtype=torch.float,device=self.device)
        cls_x_param_num, cls_y_param_num = 1, 1
        if self.extra_dense:
            cls_x=[
                self.dense_concat,
                self.res_dense_block1,
                self.res_dense_block2,
                # self.classifier,
            ]
            for model in cls_x:
                # print(model)
                for param in model.parameters():
                    cls_x_regul += torch.norm(param, lp) ** 1
                    cls_x_param_num += param.numel()

        # if self.extra_classifier:
        #     cls_y=[
        #         self.dense_concat_ab,
        #         self.res_dense_block3,
        #         self.res_dense_block4,
        #         # self.classifier_ar,
        #     ]
        #     for model in cls_y:
        #         for param in model.parameters():
        #             cls_y_regul += torch.norm(param, lp) ** 2
        #             cls_y_param_num += param.numel()
        # return [0.01*cls_x_regul/cls_x_param_num,0.01*cls_y_regul/cls_y_param_num]
        return [0.01 * total_regul / total_param_num,0.01*cls_x_regul/cls_x_param_num]

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None,attn_score=None):
        # [ag2ab_s,abl2abh_s,abh2abl_s,ab2ag_s]=attn_score
        # ab_x = self.ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
        # ab_l = self.ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])

        ab_x = torch.permute(ab_x, [0, 2, 1])
        # print(ab_x.shape)
        ab_l = torch.permute(ab_l, [0, 2, 1])

        # antibody acnn x cross attention
        ab_h1 = self.ab_gnn1(ab_x) # torch.Size([24, 200, 20])
        # print(ab_h1.shape)
        ab_l1 = self.ab_gnn1(ab_l)
        ag_h1 = self.ag_gnn1(ag_x)

        ag2ab_x,ag2ab_s = self.cross_attn_layer_ag2ab1(query=ab_h1, key=ag_h1, value=ag_h1, pos_embedding=ag_h1)
        abl2abh_x,abl2abh_s = self.cross_attn_layer_abl2h1(query=ab_h1, key=ab_l1, value=ab_l1,pos_embedding=ab_l1)
        ab_h2 = ab_h1 + ag2ab_x + abl2abh_x
        abh2abl_x,abh2abl_s = self.cross_attn_layer_abh2l1(query=ab_l1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ab_l2 = ab_l1 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag1(query=ag_h1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1)
        ag_h2 = ag_h1 + ab2ag_x

        ab_h2 = self.ab_gnn2(ab_h2)
        ab_l2 = self.ab_gnn2(ab_l2)
        ag_h2 = self.ag_gnn2(ag_h2)

        ag2ab_x, ag2ab_s = self.cross_attn_layer_ag2ab2(query=ab_h2, key=ag_h2, value=ag_h2, pos_embedding=ag_h2, edge_weights=ag2ab_s)
        abl2abh_x, abl2abh_s = self.cross_attn_layer_abl2h2(query=ab_h2, key=ab_l2, value=ab_l2, pos_embedding=ab_l2, edge_weights=abl2abh_s)
        ab_h3 = ab_h2 + ag2ab_x + abl2abh_x
        abh2abl_x, abh2abl_s = self.cross_attn_layer_abh2l2(query=ab_l2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=abh2abl_s)
        ab_l3 = ab_l2 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag2(query=ag_h2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=ab2ag_s)
        ag_h3 = ag_h2 + ab2ag_x

        ab_h3 = self.ab_gnn3(ab_h3)
        ab_l3 = self.ab_gnn3(ab_l3)
        ag_h3 = self.ag_gnn3(ag_h3)

        ab_h3 = self.expand(ab_h3)
        ab_l3 = self.expand(ab_l3)
        ag_h3 = self.expand(ag_h3)

        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h3, kernel_size=ag_h3.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h3, kernel_size=ab_h3.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l3, kernel_size=ab_l3.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, x3], 1)  # batch_size * 124

        # x = self.classifier(x)

        x = self.dense_concat(x)
        x=x + self.res_dense_block(x)
        if self.extra_dense:
            x = x + self.res_dense_block1(x)
            x = x + self.res_dense_block2(x)
        x = self.classifier(x)
        x = self.sigmoid(x)
        if self.output_attn_score:
            return x,[torch.sum(ag2ab_s,dim=1) * torch.sum(torch.permute(ab2ag_s,(0,1,3,2)),dim=1),torch.sum(torch.permute(abh2abl_s,(0,1,3,2)),dim=1) * torch.sum(abl2abh_s,dim=1)]
        else:
            return [x]

class XBCR_ACEXN_Epitope(torch.nn.Module):

    def __init__(self, node_attr_dim=1024, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=2, ab_freeze_layer_count=None,output_attn_score=False):
        super(XBCR_ACEXN_Epitope, self).__init__()
        self.output_attn_score=output_attn_score
        self.ab_bert = BertModel.from_pretrained("train_2_193999")
        # freeze the ? layers
        if ab_freeze_layer_count is not None:
            _freeze_bert(self.ab_bert, freeze_bert=False, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn3 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()

        self.cross_attn_layer_abh2l1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abl2h1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag2ab1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab2ag1 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abh2l2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_abl2h2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ag2ab2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        self.cross_attn_layer_ab2ag2 = EvolutMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)

        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
            self.ab_gnn3 = self.ag_gnn3
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn3 = torch.nn.Sequential(*atrous_blocks)

        self.dense_concat = torch.nn.Linear(3 * self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.classifier = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(3 * self.hidden_dim, self.h1_dim),
        #     self.relu,
        #     torch.nn.Linear(self.h1_dim ,self.h2_dim),
        #     self.relu,
        #     torch.nn.Dropout(dropout),
        #     torch.nn.Linear(self.h2_dim, 1))
        #

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None,attn_score=None):
        [ag2ab_s,abl2abh_s,abh2abl_s,ab2ag_s]=attn_score
        ab_x = self.ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
        ab_l = self.ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])
        ab_x = torch.permute(ab_x, [0, 2, 1])
        ab_l = torch.permute(ab_l, [0, 2, 1])

        # antibody acnn x cross attention
        ab_h1 = self.ab_gnn1(ab_x)
        ab_l1 = self.ab_gnn1(ab_l)
        ag_h1 = self.ag_gnn1(ag_x)

        ag2ab_x,ag2ab_s = self.cross_attn_layer_ag2ab1(query=ab_h1, key=ag_h1, value=ag_h1, pos_embedding=ag_h1, edge_weights=ag2ab_s)
        abl2abh_x,abl2abh_s = self.cross_attn_layer_abl2h1(query=ab_h1, key=ab_l1, value=ab_l1,pos_embedding=ab_l1, edge_weights=abl2abh_s)
        ab_h2 = ab_h1 + ag2ab_x + abl2abh_x
        abh2abl_x,abh2abl_s = self.cross_attn_layer_abh2l1(query=ab_l1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1, edge_weights=abh2abl_s)
        ab_l2 = ab_l1 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag1(query=ag_h1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1, edge_weights=ab2ag_s)
        ag_h2 = ag_h1 + ab2ag_x

        ab_h2 = self.ab_gnn2(ab_h2)
        ab_l2 = self.ab_gnn2(ab_l2)
        ag_h2 = self.ag_gnn2(ag_h2)

        ag2ab_x, ag2ab_s = self.cross_attn_layer_ag2ab2(query=ab_h2, key=ag_h2, value=ag_h2, pos_embedding=ag_h2, edge_weights=ag2ab_s)
        abl2abh_x, abl2abh_s = self.cross_attn_layer_abl2h2(query=ab_h2, key=ab_l2, value=ab_l2, pos_embedding=ab_l2, edge_weights=abl2abh_s)
        ab_h3 = ab_h2 + ag2ab_x + abl2abh_x
        abh2abl_x, abh2abl_s = self.cross_attn_layer_abh2l2(query=ab_l2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=abh2abl_s)
        ab_l3 = ab_l2 + abh2abl_x
        ab2ag_x, ab2ag_s = self.cross_attn_layer_ab2ag2(query=ag_h2, key=ab_h2, value=ab_h2, pos_embedding=ab_h2, edge_weights=ab2ag_s)
        ag_h3 = ag_h2 + ab2ag_x

        ab_h3 = self.ab_gnn3(ab_h3)
        ab_l3 = self.ab_gnn3(ab_l3)
        ag_h3 = self.ag_gnn3(ag_h3)

        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h3, kernel_size=ag_h3.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h3, kernel_size=ab_h3.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l3, kernel_size=ab_l3.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
        # x = self.classifier(x)

        x = self.dense_concat(x)
        x = self.classifier(x + self.res_dense_block(x))
        x = self.sigmoid(x)
        if self.output_attn_score:
            return x,[torch.sum(ag2ab_s,dim=1) * torch.sum(torch.permute(ab2ag_s,(0,1,3,2)),dim=1),torch.sum(torch.permute(abh2abl_s,(0,1,3,2)),dim=1) * torch.sum(abl2abh_s,dim=1)]
        else:
            return x


class XBCR_TRM(torch.nn.Module):

    def __init__(self, onehot_dim=20, node_attr_dim=20, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=True,
                 dropout=0., heads=4, block_num=5, ab_freeze_layer_count=None):
        super(XBCR_TRM, self).__init__()
        # self.ab_bert = BertModel.from_pretrained("train_2_193999")
        # if ab_freeze_layer_count is not None:
            # freeze the ? layers
            # _freeze_bert(self.ab_bert, freeze_bert=False, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        self.onehot_dim = onehot_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)

        self.ag_gnn1 = torch.nn.Linear(self.onehot_dim,self.hidden_dim)
        ffd_blocks = [FeedForward(d_model=self.hidden_dim, hidden=self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.ModuleList(ffd_blocks)
        # self.sigmoid = torch.nn.Sigmoid()
        self.sigmoid = torch.nn.Tanh()

        attn_blocks = [CrossMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.) for _ in
                       range(block_num)]
        # attn_blocks = [nn.MultiheadAttention(self.node_attr_dim, 16) for _ in range(block_num)]
        self.self_attn_layer_ab1 = torch.nn.ModuleList(attn_blocks)
        attn_blocks = [CrossMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.) for _ in
                       range(block_num)]
        # attn_blocks = [nn.MultiheadAttention(self.onehot_dim, 6) for _ in range(block_num)]
        self.self_attn_layer_ag1 = torch.nn.ModuleList(attn_blocks)
        # self.self_attn_layer_ab2 = RelativeMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
        # self.self_attn_layer_ag2 = RelativeMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)

        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
            # self.ab_gnn3 = self.ag_gnn3
        else:
            self.ab_gnn1 = FeedForward(d_model=self.node_attr_dim, hidden=self.hidden_dim)
            ffd_blocks = [FeedForward(d_model=self.node_attr_dim, hidden=self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.ModuleList(ffd_blocks)

        self.dense_concat = torch.nn.Linear(2 * self.hidden_dim + self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.classifier = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None):
        # ab_x = self.ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
        # ab_l = self.ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # ag_x = ag_x.permute([0, 2, 1])
        # ab_x = ab_x.permute([0, 2, 1])
        # ab_l = ab_l.permute( [0, 2, 1])
        # antibody acnn x cross attention
        ab_h1 = self.ab_gnn1(ab_x)
        ab_l1 = self.ab_gnn1(ab_l)
        ag_h1 = self.ag_gnn1(ag_x)

        for i in range(self.block_num):
            # print(self.self_attn_layer_ab1[i](ab_h1, ab_h1, ab_h1)[0].shape)
            # torch.Size([4, 150, 1024])
            # ab_h1 = self.ab_gnn2[i](
            #     ab_h1 + self.self_attn_layer_ab1[i](ab_h1, ab_h1, ab_h1)[0])
            # ab_l1 = self.ab_gnn2[i](
            #     ab_l1 + self.self_attn_layer_ab1[i](ab_l1, ab_l1, ab_l1)[0])
            # ag_h1 = self.ag_gnn2[i](
            #     ag_h1 + self.self_attn_layer_ag1[i](ag_h1, ag_h1, ag_h1)[0])
            ab_h1_p = ab_h1.permute([0, 2, 1])
            ab_l1_p = ab_l1.permute([0, 2, 1])
            ag_h1_p = ag_h1.permute([0, 2, 1])
            ab_ha = self.self_attn_layer_ab1[i](query=ab_h1_p, key=ab_h1_p, value=ab_h1_p, pos_embedding=ab_h1_p)
            ab_la = self.self_attn_layer_ab1[i](query=ab_l1_p, key=ab_l1_p, value=ab_l1_p, pos_embedding=ab_l1_p)
            ag_aa = self.self_attn_layer_ag1[i](query=ag_h1_p, key=ag_h1_p, value=ag_h1_p, pos_embedding=ag_h1_p)

            ab_ha = ab_ha.permute([0, 2, 1])
            ab_la = ab_la.permute([0, 2, 1])
            ag_aa = ag_aa.permute([0, 2, 1])

            ab_h1 = self.ab_gnn2[i](ab_h1 + ab_ha)
            ab_l1 = self.ab_gnn2[i](ab_l1 + ab_la)
            ag_h1 = self.ag_gnn2[i](ag_h1 + ag_aa)

        # print(ab_h1.shape)
        # torch.Size([4, 150, 1024])
        # print(ab_l1.shape)
        # torch.Size([4, 150, 1024])
        # print(ag_h1.shape)
        # torch.Size([4, 800, 30])
        # ag_h1=torch.permute(ag_h1,[0,2,1])
        # ab_h1=torch.permute(ab_h1,[0,2,1])
        # ab_l1=torch.permute(ab_l1,[0,2,1])
        ag_h1 = ag_h1.permute([0, 2, 1])
        ab_h1 = ab_h1.permute([0, 2, 1])
        ab_l1 = ab_l1.permute( [0, 2, 1])
        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h1, kernel_size=ag_h1.size()[2])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h1, kernel_size=ab_h1.size()[2])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l1, kernel_size=ab_l1.size()[2])[..., 0]

        # print(x1.shape)
        # torch.Size([4, 30])
        # print(x2.shape)
        # torch.Size([4, 1024])
        # print(x3.shape)
        # torch.Size([4, 1024])
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
        # x = self.classifier(x)
        # print(x.shape)
        # torch.Size([4, 2078])
        x = self.dense_concat(x)
        # print(x.shape)
        # torch.Size([4, 64])
        x = self.classifier(x + self.res_dense_block(x))
        # print(x)
        # print(x.shape)
        # torch.Size([4, 1])
        x = self.sigmoid(x)
        # x = self.sigmoid(x)

        return [x]


class RESNET(torch.nn.Module):
    def __init__(self):
        super(RESNET, self).__init__()
        self.model = torchvision.models.resnet50()
        # self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth'))
        self.model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = torch.nn.Linear(2048, 512)
        self.fc2 = torch.nn.Linear(512, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, ag_x, ab_x, ab_l):
        ag_x = ag_x.view(ag_x.size()[0], 1, ag_x.size()[1], ag_x.size()[2])
        ab_x = ab_x.view(ab_x.size()[0], 1, ab_x.size()[1], ab_x.size()[2])
        ab_l = ab_l.view(ab_l.size()[0], 1, ab_l.size()[1], ab_l.size()[2])
        x = torch.cat([ag_x, ab_x, ab_l], dim=1)

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc1(x)
        x = self.model.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return [x]

# class XBCR_TRM(torch.nn.Module):
#
#     def __init__(self, onehot_dim=30, node_attr_dim=1024, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=False,
#                  dropout=0., heads=4, block_num=4, ab_freeze_layer_count=None):
#         super(XBCR_TRM, self).__init__()
#         self.ab_bert = BertModel.from_pretrained("train_2_193999")
#         if ab_freeze_layer_count is not None:
#             # freeze the ? layers
#             _freeze_bert(self.ab_bert, freeze_bert=False, freeze_layer_count=ab_freeze_layer_count)
#
#         self.node_attr_dim = node_attr_dim
#         self.onehot_dim = onehot_dim
#         # self.edge_attr_dim = edge_attr_dim
#         self.hidden_dim = hidden_dim
#         self.h1_dim = h1_dim  # dim after the first gnn
#         self.h2_dim = h2_dim  # dim after the bipartite
#         self.share_weight = share_weight
#         self.n_dropout = dropout
#         self.heads = heads
#         self.block_num = block_num
#         self.relu = torch.nn.LeakyReLU()
#         # self.gmp = torch.nn.MaxPool1d()
#         self.dropout = torch.nn.Dropout(dropout)
#         # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
#
#         self.ag_gnn1 = FeedForward(d_model=self.onehot_dim, hidden=self.hidden_dim)
#         ffd_blocks = [FeedForward(d_model=self.onehot_dim, hidden=self.hidden_dim) for _ in range(block_num)]
#         self.ag_gnn2 = torch.nn.ModuleList(ffd_blocks)
#         # self.sigmoid = torch.nn.Sigmoid()
#         self.sigmoid = torch.nn.Tanh()
#
#         attn_blocks = [CrossMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.) for _ in
#                        range(block_num)]
#         # attn_blocks = [nn.MultiheadAttention(self.node_attr_dim, 16) for _ in range(block_num)]
#         self.self_attn_layer_ab1 = torch.nn.ModuleList(attn_blocks)
#         attn_blocks = [CrossMultiHeadAttention(d_model=self.onehot_dim, num_heads=6, dropout_p=0.) for _ in
#                        range(block_num)]
#         # attn_blocks = [nn.MultiheadAttention(self.onehot_dim, 6) for _ in range(block_num)]
#         self.self_attn_layer_ag1 = torch.nn.ModuleList(attn_blocks)
#         # self.self_attn_layer_ab2 = RelativeMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
#         # self.self_attn_layer_ag2 = RelativeMultiHeadAttention(d_model=self.hidden_dim, num_heads=16, dropout_p=0.)
#
#         if self.share_weight:
#             self.ab_gnn1 = self.ag_gnn1
#             self.ab_gnn2 = self.ag_gnn2
#             # self.ab_gnn3 = self.ag_gnn3
#         else:
#             self.ab_gnn1 = FeedForward(d_model=self.node_attr_dim, hidden=self.hidden_dim)
#             ffd_blocks = [FeedForward(d_model=self.node_attr_dim, hidden=self.hidden_dim) for _ in range(block_num)]
#             self.ab_gnn2 = torch.nn.ModuleList(ffd_blocks)
#
#         self.dense_concat = torch.nn.Linear(2 * self.node_attr_dim + self.onehot_dim, self.h1_dim)
#         self.res_dense_block = torch.nn.Sequential(
#             self.relu,
#             torch.nn.Linear(self.h1_dim, self.h1_dim),
#             self.relu,
#             torch.nn.Linear(self.h1_dim, self.h1_dim),
#             self.relu,
#             torch.nn.Dropout(dropout),
#             torch.nn.Linear(self.h1_dim, self.h1_dim))
#         self.classifier = torch.nn.Sequential(
#             self.relu,
#             torch.nn.Linear(self.h1_dim, 1))
#
#         # self.classifier = torch.nn.Sequential(
#         #     torch.nn.Linear(3 * self.hidden_dim, self.h1_dim),
#         #     self.relu,
#         #     torch.nn.Linear(self.h1_dim ,self.h2_dim),
#         #     self.relu,
#         #     torch.nn.Dropout(dropout),
#         #     torch.nn.Linear(self.h2_dim, 1))
#
#     def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
#                 attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None):
#         ab_x = self.ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
#         ab_l = self.ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
#         # print(ab_x.shape)
#         # torch.Size([4, 150, 1024])
#         # print(ab_l.shape)
#         # torch.Size([4, 150, 1024])
#         # print(ag_x.shape)
#         # torch.Size([4, 800, 30])
#
#         # antigen acnn
#         # print(ag_x)
#         # print(ag_edge_index)
#         # ag_x=torch.permute(ag_x,[0,2,1])
#         # ab_x=torch.permute(ab_x,[0,2,1])
#         # ab_l=torch.permute(ab_l,[0,2,1])
#
#         # antibody acnn x cross attention
#         ab_h1 = self.ab_gnn1(ab_x)
#         ab_l1 = self.ab_gnn1(ab_l)
#         ag_h1 = self.ag_gnn1(ag_x)
#         # print(ab_h1.shape)
#         # torch.Size([4, 150, 1024])
#         # print(ab_l1.shape)
#         # torch.Size([4, 150, 1024])
#         # print(ag_h1.shape)
#         # torch.Size([4, 800, 30])
#
#         for i in range(self.block_num):
#             # print(self.self_attn_layer_ab1[i](ab_h1, ab_h1, ab_h1)[0].shape)
#             # torch.Size([4, 150, 1024])
#             # ab_h1 = self.ab_gnn2[i](
#             #     ab_h1 + self.self_attn_layer_ab1[i](ab_h1, ab_h1, ab_h1)[0])
#             # ab_l1 = self.ab_gnn2[i](
#             #     ab_l1 + self.self_attn_layer_ab1[i](ab_l1, ab_l1, ab_l1)[0])
#             # ag_h1 = self.ag_gnn2[i](
#             #     ag_h1 + self.self_attn_layer_ag1[i](ag_h1, ag_h1, ag_h1)[0])
#             ab_h1 = self.ab_gnn2[i](
#                 ab_h1 + self.self_attn_layer_ab1[i](query=ab_h1, key=ab_h1, value=ab_h1, pos_embedding=ab_h1))
#             ab_l1 = self.ab_gnn2[i](
#                 ab_l1 + self.self_attn_layer_ab1[i](query=ab_l1, key=ab_l1, value=ab_l1, pos_embedding=ab_l1))
#             ag_h1 = self.ag_gnn2[i](
#                 ag_h1 + self.self_attn_layer_ag1[i](query=ag_h1, key=ag_h1, value=ag_h1, pos_embedding=ag_h1))
#
#         # print(ab_h1.shape)
#         # torch.Size([4, 150, 1024])
#         # print(ab_l1.shape)
#         # torch.Size([4, 150, 1024])
#         # print(ag_h1.shape)
#         # torch.Size([4, 800, 30])
#         ag_h1=torch.permute(ag_h1,[0,2,1])
#         ab_h1=torch.permute(ab_h1,[0,2,1])
#         ab_l1=torch.permute(ab_l1,[0,2,1])
#         # predict bind or not
#         x1 = torch.nn.functional.max_pool1d(ag_h1, kernel_size=ag_h1.size()[2])[..., 0]
#         x2 = torch.nn.functional.max_pool1d(ab_h1, kernel_size=ab_h1.size()[2])[..., 0]
#         x3 = torch.nn.functional.max_pool1d(ab_l1, kernel_size=ab_l1.size()[2])[..., 0]
#
#         # print(x1.shape)
#         # torch.Size([4, 30])
#         # print(x2.shape)
#         # torch.Size([4, 1024])
#         # print(x3.shape)
#         # torch.Size([4, 1024])
#         # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
#         x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
#         # x = self.classifier(x)
#         # print(x.shape)
#         # torch.Size([4, 2078])
#         x = self.dense_concat(x)
#         # print(x.shape)
#         # torch.Size([4, 64])
#         x = self.classifier(x + self.res_dense_block(x))
#         # print(x)
#         # print(x.shape)
#         # torch.Size([4, 1])
#         x = self.sigmoid(x)
#         # x = self.sigmoid(x)

        # return [x]

# lm_gnn = DeepBCR_ACEXN()

# ======================================================================================================================
# ReptileModel
# ======================================================================================================================
class ReptileModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def point_grad_to(self, target):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if self.is_cuda():
                    p.grad = torch.autograd.Variable(torch.zeros(p.size())).cuda()
                else:
                    p.grad = torch.autograd.Variable(torch.zeros(p.size()))
            p.grad.data.zero_()  # not sure this is required
            p.grad.data.add_(p.data - target_p.data)

    def is_cuda(self):
        return next(self.parameters()).is_cuda


class OmniglotModel(ReptileModel):
    """
    A model for Omniglot classification.
    """
    def __init__(self, num_classes):
        ReptileModel.__init__(self)

        self.num_classes = num_classes

        self.conv = nn.Sequential(
            # 28 x 28 - 1
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 14 x 14 - 64
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 7 x 7 - 64
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 4 x 4 - 64
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 2 x 2 - 64
        )

        self.classifier = nn.Sequential(
            # 2 x 2 x 64 = 256
            nn.Linear(256, num_classes),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        out = x.view(-1, 1, 28, 28)
        out = self.conv(out)
        out = out.view(len(out), -1)
        out = self.classifier(out)
        return out

    def predict(self, prob):
        __, argmax = prob.max(1)
        return argmax

    def clone(self):
        clone = OmniglotModel(self.num_classes)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone

class XBCR_ACNN_meta(ReptileModel):

    def __init__(self, node_attr_dim=1024, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=3, freeze_bert=False, ab_freeze_layer_count=None, extra_dense=False):
        super(ReptileModel, self).__init__()
        # self.ab_bert = BertModel.from_pretrained("train_2_193999")
        # self.ab_bert = BertModel.from_pretrained("prot_bert")
        # freeze the ? layers
        self.freeze_bert=freeze_bert
        self.ab_freeze_layer_count=ab_freeze_layer_count
        # if ab_freeze_layer_count is not None:
        #     _freeze_bert(self.ab_bert, freeze_bert=freeze_bert, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()
        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)

        self.dense_concat = torch.nn.Linear(3 * self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.classifier = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

    def clone_meta(self):
        clone = XBCR_ACNN_meta(node_attr_dim=self.node_attr_dim, hidden_dim=self.hidden_dim, h1_dim=self.h1_dim, h2_dim=self.h2_dim, share_weight=self.share_weight, dropout=self.n_dropout,
                 heads=self.heads, block_num=self.block_num, freeze_bert=self.freeze_bert, ab_freeze_layer_count=self.ab_freeze_layer_count)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone

    def get_variables(self,lp=1):
        self.device=next(self.dense_concat.parameters()).device
        total_regul=torch.tensor(0,dtype=torch.float,device=self.device)
        total_param_num = 1
        for param in self.parameters():
            total_regul += torch.norm(param, lp) ** 1
            total_param_num += param.numel()
        cls_x_regul,cls_y_regul=torch.tensor(0,dtype=torch.float,device=self.device),torch.tensor(0,dtype=torch.float,device=self.device)
        cls_x_param_num, cls_y_param_num = 1, 1
        # if self.extra_dense:
        #     cls_x=[
        #         self.dense_concat,
        #         self.res_dense_block1,
        #         self.res_dense_block2,
        #         # self.classifier,
        #     ]
        #     for model in cls_x:
        #         # print(model)
        #         for param in model.parameters():
        #             cls_x_regul += torch.norm(param, lp) ** 1
        #             cls_x_param_num += param.numel()

        # if self.extra_classifier:
        #     cls_y=[
        #         self.dense_concat_ab,
        #         self.res_dense_block3,
        #         self.res_dense_block4,
        #         # self.classifier_ar,
        #     ]
        #     for model in cls_y:
        #         for param in model.parameters():
        #             cls_y_regul += torch.norm(param, lp) ** 2
        #             cls_y_param_num += param.numel()
        # return [0.01*cls_x_regul/cls_x_param_num,0.01*cls_y_regul/cls_y_param_num]
        return [0.01 * total_regul / total_param_num,0.01*cls_x_regul/cls_x_param_num]

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None,ab_bert=None):

        # if ab_bert is not None:
        if True:
            ab_x = ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
            ab_l = ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])
        ab_x = torch.permute(ab_x, [0, 2, 1])
        ab_l = torch.permute(ab_l, [0, 2, 1])

        ag_h1 = self.ag_gnn1(ag_x)
        ag_h1 = self.ag_gnn2(ag_h1)

        # antibody acnn
        ab_h1 = self.ab_gnn1(ab_x)
        ab_h1 = self.ab_gnn2(ab_h1)

        ab_l1 = self.ab_gnn1(ab_l)
        ab_l1 = self.ab_gnn2(ab_l1)

        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h1, kernel_size=ag_h1.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h1, kernel_size=ab_h1.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l1, kernel_size=ab_l1.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
        x = self.dense_concat(x)
        x = self.classifier(x + self.res_dense_block(x))
        x = self.sigmoid(x)
        # x = torch.clamp(x, min=0.001, max=0.999)
        return [x]

class XBCR_ACNN_dense_meta_clamp(ReptileModel):

    def __init__(self, node_attr_dim=1024, hidden_dim=128, h1_dim=256, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=2, freeze_bert=False, ab_freeze_layer_count=None, extra_dense=True):
        super(ReptileModel, self).__init__()
        expand_num = 4
        # self.ab_bert = BertModel.from_pretrained("train_2_193999")
        # self.ab_bert = BertModel.from_pretrained("prot_bert")
        # freeze the ? layers
        self.freeze_bert=freeze_bert
        self.ab_freeze_layer_count=ab_freeze_layer_count
        # if ab_freeze_layer_count is not None:
        #     _freeze_bert(self.ab_bert, freeze_bert=freeze_bert, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.extra_dense=extra_dense
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()
        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)

        self.expand = Atrous_block(self.hidden_dim, expand_num * self.hidden_dim)

        self.dense_concat = torch.nn.Linear(3 * expand_num * self.hidden_dim, self.h1_dim)

        if self.extra_dense:
            # self.h1_dim=128
            self.res_dense_block1 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
            self.res_dense_block2 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.classifier = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

    def clone_meta(self):
        clone = XBCR_ACNN_dense_meta_clamp(node_attr_dim=self.node_attr_dim, hidden_dim=self.hidden_dim, h1_dim=self.h1_dim, h2_dim=self.h2_dim, share_weight=self.share_weight, dropout=self.n_dropout,
                 heads=self.heads, block_num=self.block_num, freeze_bert=self.freeze_bert, ab_freeze_layer_count=self.ab_freeze_layer_count)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone

    def get_variables(self,lp=1):
        self.device=next(self.dense_concat.parameters()).device
        total_regul=torch.tensor(0,dtype=torch.float,device=self.device)
        total_param_num = 1
        for param in self.parameters():
            total_regul += torch.norm(param, lp) ** 1
            total_param_num += param.numel()
        cls_x_regul,cls_y_regul=torch.tensor(0,dtype=torch.float,device=self.device),torch.tensor(0,dtype=torch.float,device=self.device)
        cls_x_param_num, cls_y_param_num = 1, 1
        # if self.extra_dense:
        #     cls_x=[
        #         self.dense_concat,
        #         self.res_dense_block1,
        #         self.res_dense_block2,
        #         # self.classifier,
        #     ]
        #     for model in cls_x:
        #         # print(model)
        #         for param in model.parameters():
        #             cls_x_regul += torch.norm(param, lp) ** 1
        #             cls_x_param_num += param.numel()

        # if self.extra_classifier:
        #     cls_y=[
        #         self.dense_concat_ab,
        #         self.res_dense_block3,
        #         self.res_dense_block4,
        #         # self.classifier_ar,
        #     ]
        #     for model in cls_y:
        #         for param in model.parameters():
        #             cls_y_regul += torch.norm(param, lp) ** 2
        #             cls_y_param_num += param.numel()
        # return [0.01*cls_x_regul/cls_x_param_num,0.01*cls_y_regul/cls_y_param_num]
        return [0.01 * total_regul / total_param_num,0.01*cls_x_regul/cls_x_param_num]

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None,ab_bert=None):

        # if ab_bert is not None:
        if True:
            ab_x = ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
            ab_l = ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])
        ab_x = torch.permute(ab_x, [0, 2, 1])
        ab_l = torch.permute(ab_l, [0, 2, 1])

        ag_h1 = self.ag_gnn1(ag_x)
        ag_h1 = self.ag_gnn2(ag_h1)

        # antibody acnn
        ab_h1 = self.ab_gnn1(ab_x)
        ab_h1 = self.ab_gnn2(ab_h1)

        ab_l1 = self.ab_gnn1(ab_l)
        ab_l1 = self.ab_gnn2(ab_l1)

        ab_h1 = self.expand(ab_h1)
        ab_l1 = self.expand(ab_l1)
        ag_h1 = self.expand(ag_h1)
        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h1, kernel_size=ag_h1.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h1, kernel_size=ab_h1.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l1, kernel_size=ab_l1.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
        x = self.dense_concat(x)
        if self.extra_dense:
            x = x + self.res_dense_block1(x)
            x = x + self.res_dense_block2(x)
        x = self.classifier(x + self.res_dense_block(x))
        x = self.sigmoid(x)
        x = torch.clamp(x, min=0.001, max=0.999)
        return [x]


class XBCR_ACNN_dense_meta(ReptileModel):

    def __init__(self, node_attr_dim=1024, hidden_dim=128, h1_dim=256, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=2, freeze_bert=False, ab_freeze_layer_count=None, extra_dense=True):
        super(ReptileModel, self).__init__()
        expand_num = 4
        # self.ab_bert = BertModel.from_pretrained("train_2_193999")
        # self.ab_bert = BertModel.from_pretrained("prot_bert")
        # freeze the ? layers
        self.freeze_bert=freeze_bert
        self.ab_freeze_layer_count=ab_freeze_layer_count
        # if ab_freeze_layer_count is not None:
        #     _freeze_bert(self.ab_bert, freeze_bert=freeze_bert, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.extra_dense=extra_dense
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()
        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)

        self.expand = Atrous_block(self.hidden_dim, expand_num * self.hidden_dim)

        self.dense_concat = torch.nn.Linear(3 * expand_num * self.hidden_dim, self.h1_dim)

        if self.extra_dense:
            # self.h1_dim=128
            self.res_dense_block1 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
            self.res_dense_block2 = torch.nn.Sequential(
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                torch.nn.Linear(self.h1_dim, self.h1_dim),
                self.relu,
                # torch.nn.Dropout(dropout),
                torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.classifier = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

    def clone_meta(self):
        clone = XBCR_ACNN_dense_meta(node_attr_dim=self.node_attr_dim, hidden_dim=self.hidden_dim, h1_dim=self.h1_dim, h2_dim=self.h2_dim, share_weight=self.share_weight, dropout=self.n_dropout,
                 heads=self.heads, block_num=self.block_num, freeze_bert=self.freeze_bert, ab_freeze_layer_count=self.ab_freeze_layer_count)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone

    def get_variables(self,lp=1):
        self.device=next(self.dense_concat.parameters()).device
        total_regul=torch.tensor(0,dtype=torch.float,device=self.device)
        total_param_num = 1
        for param in self.parameters():
            total_regul += torch.norm(param, lp) ** 1
            total_param_num += param.numel()
        cls_x_regul,cls_y_regul=torch.tensor(0,dtype=torch.float,device=self.device),torch.tensor(0,dtype=torch.float,device=self.device)
        cls_x_param_num, cls_y_param_num = 1, 1
        # if self.extra_dense:
        #     cls_x=[
        #         self.dense_concat,
        #         self.res_dense_block1,
        #         self.res_dense_block2,
        #         # self.classifier,
        #     ]
        #     for model in cls_x:
        #         # print(model)
        #         for param in model.parameters():
        #             cls_x_regul += torch.norm(param, lp) ** 1
        #             cls_x_param_num += param.numel()

        # if self.extra_classifier:
        #     cls_y=[
        #         self.dense_concat_ab,
        #         self.res_dense_block3,
        #         self.res_dense_block4,
        #         # self.classifier_ar,
        #     ]
        #     for model in cls_y:
        #         for param in model.parameters():
        #             cls_y_regul += torch.norm(param, lp) ** 2
        #             cls_y_param_num += param.numel()
        # return [0.01*cls_x_regul/cls_x_param_num,0.01*cls_y_regul/cls_y_param_num]
        return [0.01 * total_regul / total_param_num,0.01*cls_x_regul/cls_x_param_num]

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None,ab_bert=None):

        # if ab_bert is not None:
        if True:
            ab_x = ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
            ab_l = ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])
        ab_x = torch.permute(ab_x, [0, 2, 1])
        ab_l = torch.permute(ab_l, [0, 2, 1])

        ag_h1 = self.ag_gnn1(ag_x)
        ag_h1 = self.ag_gnn2(ag_h1)

        # antibody acnn
        ab_h1 = self.ab_gnn1(ab_x)
        ab_h1 = self.ab_gnn2(ab_h1)

        ab_l1 = self.ab_gnn1(ab_l)
        ab_l1 = self.ab_gnn2(ab_l1)

        ab_h1 = self.expand(ab_h1)
        ab_l1 = self.expand(ab_l1)
        ag_h1 = self.expand(ag_h1)
        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h1, kernel_size=ag_h1.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h1, kernel_size=ab_h1.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l1, kernel_size=ab_l1.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
        x = self.dense_concat(x)
        if self.extra_dense:
            x = x + self.res_dense_block1(x)
            x = x + self.res_dense_block2(x)
        x = self.classifier(x + self.res_dense_block(x))
        x = self.sigmoid(x)
        # x = torch.clamp(x, min=0.001, max=0.999)
        return [x]


class XBCR_ACNN_meta_concat(ReptileModel):

    def __init__(self, node_attr_dim=1024, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=3, freeze_bert=False, ab_freeze_layer_count=None, extra_dense=False):
        super(ReptileModel, self).__init__()
        # self.ab_bert = BertModel.from_pretrained("train_2_193999")
        # self.ab_bert = BertModel.from_pretrained("prot_bert")
        # freeze the ? layers
        self.freeze_bert=freeze_bert
        self.ab_freeze_layer_count=ab_freeze_layer_count
        # if ab_freeze_layer_count is not None:
        #     _freeze_bert(self.ab_bert, freeze_bert=freeze_bert, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()
        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)

        self.dense_concat = torch.nn.Linear(2 * self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.classifier = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

    def clone_meta(self):
        clone = XBCR_ACNN_meta(node_attr_dim=self.node_attr_dim, hidden_dim=self.hidden_dim, h1_dim=self.h1_dim, h2_dim=self.h2_dim, share_weight=self.share_weight, dropout=self.n_dropout,
                 heads=self.heads, block_num=self.block_num, freeze_bert=self.freeze_bert, ab_freeze_layer_count=self.ab_freeze_layer_count)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone

    def get_variables(self,lp=1):
        self.device=next(self.dense_concat.parameters()).device
        total_regul=torch.tensor(0,dtype=torch.float,device=self.device)
        total_param_num = 1
        for param in self.parameters():
            total_regul += torch.norm(param, lp) ** 1
            total_param_num += param.numel()
        cls_x_regul,cls_y_regul=torch.tensor(0,dtype=torch.float,device=self.device),torch.tensor(0,dtype=torch.float,device=self.device)
        cls_x_param_num, cls_y_param_num = 1, 1
        # if self.extra_dense:
        #     cls_x=[
        #         self.dense_concat,
        #         self.res_dense_block1,
        #         self.res_dense_block2,
        #         # self.classifier,
        #     ]
        #     for model in cls_x:
        #         # print(model)
        #         for param in model.parameters():
        #             cls_x_regul += torch.norm(param, lp) ** 1
        #             cls_x_param_num += param.numel()

        # if self.extra_classifier:
        #     cls_y=[
        #         self.dense_concat_ab,
        #         self.res_dense_block3,
        #         self.res_dense_block4,
        #         # self.classifier_ar,
        #     ]
        #     for model in cls_y:
        #         for param in model.parameters():
        #             cls_y_regul += torch.norm(param, lp) ** 2
        #             cls_y_param_num += param.numel()
        # return [0.01*cls_x_regul/cls_x_param_num,0.01*cls_y_regul/cls_y_param_num]
        return [0.01 * total_regul / total_param_num,0.01*cls_x_regul/cls_x_param_num]

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None,ab_bert=None):

        # if ab_bert is not None:
        if True:
            ab_x = ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab).last_hidden_state

        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])
        ab_x = torch.permute(ab_x, [0, 2, 1])

        ag_h1 = self.ag_gnn1(ag_x)
        ag_h1 = self.ag_gnn2(ag_h1)

        # antibody acnn
        ab_h1 = self.ab_gnn1(ab_x)
        ab_h1 = self.ab_gnn2(ab_h1)


        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h1, kernel_size=ag_h1.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h1, kernel_size=ab_h1.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, ], 1)  # batch_size * 124
        x = self.dense_concat(x)
        x = self.classifier(x + self.res_dense_block(x))
        x = self.sigmoid(x)
        x = torch.clamp(x,min=0.001,max=0.999)
        return [x]


class XBCR_ACNN_meta_diffACNN(ReptileModel):

    def __init__(self, node_attr_dim=1024, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=3, freeze_bert=False, ab_freeze_layer_count=None, extra_dense=False):
        super(ReptileModel, self).__init__()
        # self.ab_bert = BertModel.from_pretrained("train_2_193999")
        # self.ab_bert = BertModel.from_pretrained("prot_bert")
        # freeze the ? layers
        self.freeze_bert=freeze_bert
        self.ab_freeze_layer_count=ab_freeze_layer_count
        # if ab_freeze_layer_count is not None:
        #     _freeze_bert(self.ab_bert, freeze_bert=freeze_bert, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1_hiv = Atrous_block(20, self.hidden_dim)
        atrous_blocks_hiv = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2_hiv = torch.nn.Sequential(*atrous_blocks_hiv)

        self.ag_gnn1_rbd = Atrous_block(20, self.hidden_dim)
        atrous_blocks_rbd = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2_rbd = torch.nn.Sequential(*atrous_blocks_rbd)

        self.ag_gnn1_flu = Atrous_block(20, self.hidden_dim)
        atrous_blocks_flu = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2_flu = torch.nn.Sequential(*atrous_blocks_flu)

        self.sigmoid = torch.nn.Sigmoid()
        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1_hiv
            self.ab_gnn2 = self.ag_gnn2_hiv
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)

        self.dense_concat = torch.nn.Linear(3 * self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.classifier = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

    def clone_meta(self):
        clone = XBCR_ACNN_meta(node_attr_dim=self.node_attr_dim, hidden_dim=self.hidden_dim, h1_dim=self.h1_dim, h2_dim=self.h2_dim, share_weight=self.share_weight, dropout=self.n_dropout,
                 heads=self.heads, block_num=self.block_num, freeze_bert=self.freeze_bert, ab_freeze_layer_count=self.ab_freeze_layer_count)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone

    def get_variables(self,lp=1):
        self.device=next(self.dense_concat.parameters()).device
        total_regul=torch.tensor(0,dtype=torch.float,device=self.device)
        total_param_num = 1
        for param in self.parameters():
            total_regul += torch.norm(param, lp) ** 1
            total_param_num += param.numel()
        cls_x_regul,cls_y_regul=torch.tensor(0,dtype=torch.float,device=self.device),torch.tensor(0,dtype=torch.float,device=self.device)
        cls_x_param_num, cls_y_param_num = 1, 1
        # if self.extra_dense:
        #     cls_x=[
        #         self.dense_concat,
        #         self.res_dense_block1,
        #         self.res_dense_block2,
        #         # self.classifier,
        #     ]
        #     for model in cls_x:
        #         # print(model)
        #         for param in model.parameters():
        #             cls_x_regul += torch.norm(param, lp) ** 1
        #             cls_x_param_num += param.numel()

        # if self.extra_classifier:
        #     cls_y=[
        #         self.dense_concat_ab,
        #         self.res_dense_block3,
        #         self.res_dense_block4,
        #         # self.classifier_ar,
        #     ]
        #     for model in cls_y:
        #         for param in model.parameters():
        #             cls_y_regul += torch.norm(param, lp) ** 2
        #             cls_y_param_num += param.numel()
        # return [0.01*cls_x_regul/cls_x_param_num,0.01*cls_y_regul/cls_y_param_num]
        return [0.01 * total_regul / total_param_num,0.01*cls_x_regul/cls_x_param_num]

    def forward(self, ag_x, ag_type, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None,ab_bert=None):

        # if ab_bert is not None:
        if True:
            ab_x = ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
            ab_l = ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])
        ab_x = torch.permute(ab_x, [0, 2, 1])
        ab_l = torch.permute(ab_l, [0, 2, 1])
        if ag_type == 'hiv':
            ag_h1 = self.ag_gnn1_hiv(ag_x)
            ag_h1 = self.ag_gnn2_hiv(ag_h1)
        elif ag_type == 'rbd':
            ag_h1 = self.ag_gnn1_rbd(ag_x)
            ag_h1 = self.ag_gnn2_rbd(ag_h1)
        else:
            ag_h1 = self.ag_gnn1_flu(ag_x)
            ag_h1 = self.ag_gnn2_flu(ag_h1)

        # antibody acnn
        ab_h1 = self.ab_gnn1(ab_x)
        ab_h1 = self.ab_gnn2(ab_h1)

        ab_l1 = self.ab_gnn1(ab_l)
        ab_l1 = self.ab_gnn2(ab_l1)

        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h1, kernel_size=ag_h1.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h1, kernel_size=ab_h1.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l1, kernel_size=ab_l1.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
        x = self.dense_concat(x)
        x = self.classifier(x + self.res_dense_block(x))
        x = self.sigmoid(x)
        x = torch.clamp(x, min=0.001, max=0.999)
        return [x]

class XBCR_ACNN_meta_auto(ReptileModel):

    def __init__(self, node_attr_dim=1024, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=3, freeze_bert=False, ab_freeze_layer_count=None, extra_dense=False):
        super(ReptileModel, self).__init__()
        # self.ab_bert = BertModel.from_pretrained("train_2_193999")
        # self.ab_bert = BertModel.from_pretrained("prot_bert")
        # freeze the ? layers
        self.freeze_bert=freeze_bert
        self.ab_freeze_layer_count=ab_freeze_layer_count
        # if ab_freeze_layer_count is not None:
        #     _freeze_bert(self.ab_bert, freeze_bert=freeze_bert, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()
        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)

        self.dense_concat = torch.nn.Linear(2 * self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.classifier = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

    def clone_meta(self):
        clone = XBCR_ACNN_meta(node_attr_dim=self.node_attr_dim, hidden_dim=self.hidden_dim, h1_dim=self.h1_dim, h2_dim=self.h2_dim, share_weight=self.share_weight, dropout=self.n_dropout,
                 heads=self.heads, block_num=self.block_num, freeze_bert=self.freeze_bert, ab_freeze_layer_count=self.ab_freeze_layer_count)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone

    def get_variables(self,lp=1):
        self.device=next(self.dense_concat.parameters()).device
        total_regul=torch.tensor(0,dtype=torch.float,device=self.device)
        total_param_num = 1
        for param in self.parameters():
            total_regul += torch.norm(param, lp) ** 1
            total_param_num += param.numel()
        cls_x_regul,cls_y_regul=torch.tensor(0,dtype=torch.float,device=self.device),torch.tensor(0,dtype=torch.float,device=self.device)
        cls_x_param_num, cls_y_param_num = 1, 1
        # if self.extra_dense:
        #     cls_x=[
        #         self.dense_concat,
        #         self.res_dense_block1,
        #         self.res_dense_block2,
        #         # self.classifier,
        #     ]
        #     for model in cls_x:
        #         # print(model)
        #         for param in model.parameters():
        #             cls_x_regul += torch.norm(param, lp) ** 1
        #             cls_x_param_num += param.numel()

        # if self.extra_classifier:
        #     cls_y=[
        #         self.dense_concat_ab,
        #         self.res_dense_block3,
        #         self.res_dense_block4,
        #         # self.classifier_ar,
        #     ]
        #     for model in cls_y:
        #         for param in model.parameters():
        #             cls_y_regul += torch.norm(param, lp) ** 2
        #             cls_y_param_num += param.numel()
        # return [0.01*cls_x_regul/cls_x_param_num,0.01*cls_y_regul/cls_y_param_num]
        return [0.01 * total_regul / total_param_num,0.01*cls_x_regul/cls_x_param_num]

    def forward(self, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None,ab_bert=None):

        # if ab_bert is not None:
        if True:
            ab_x = ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
            ab_l = ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)

        ab_x = torch.permute(ab_x, [0, 2, 1])
        ab_l = torch.permute(ab_l, [0, 2, 1])

        # antibody acnn
        ab_h1 = self.ab_gnn1(ab_x)
        ab_h1 = self.ab_gnn2(ab_h1)

        ab_l1 = self.ab_gnn1(ab_l)
        ab_l1 = self.ab_gnn2(ab_l1)

        # predict bind or not

        x2 = torch.nn.functional.max_pool1d(ab_h1, kernel_size=ab_h1.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l1, kernel_size=ab_l1.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x2, x3], 1)  # batch_size * 124
        x = self.dense_concat(x)
        x = self.classifier(x + self.res_dense_block(x))
        x = self.sigmoid(x)
        x = torch.clamp(x, min=0.001, max=0.999)
        return [x]

class XBCR_ACNN_meta_auto_concat(ReptileModel):

    def __init__(self, node_attr_dim=1024, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=3, freeze_bert=False, ab_freeze_layer_count=None, extra_dense=False):
        super(ReptileModel, self).__init__()
        # self.ab_bert = BertModel.from_pretrained("train_2_193999")
        # self.ab_bert = BertModel.from_pretrained("prot_bert")
        # freeze the ? layers
        self.freeze_bert=freeze_bert
        self.ab_freeze_layer_count=ab_freeze_layer_count
        # if ab_freeze_layer_count is not None:
        #     _freeze_bert(self.ab_bert, freeze_bert=freeze_bert, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()
        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)

        self.dense_concat = torch.nn.Linear(1 * self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.classifier = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

    def clone_meta(self):
        clone = XBCR_ACNN_meta(node_attr_dim=self.node_attr_dim, hidden_dim=self.hidden_dim, h1_dim=self.h1_dim, h2_dim=self.h2_dim, share_weight=self.share_weight, dropout=self.n_dropout,
                 heads=self.heads, block_num=self.block_num, freeze_bert=self.freeze_bert, ab_freeze_layer_count=self.ab_freeze_layer_count)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone

    def get_variables(self,lp=1):
        self.device=next(self.dense_concat.parameters()).device
        total_regul=torch.tensor(0,dtype=torch.float,device=self.device)
        total_param_num = 1
        for param in self.parameters():
            total_regul += torch.norm(param, lp) ** 1
            total_param_num += param.numel()
        cls_x_regul,cls_y_regul=torch.tensor(0,dtype=torch.float,device=self.device),torch.tensor(0,dtype=torch.float,device=self.device)
        cls_x_param_num, cls_y_param_num = 1, 1
        # if self.extra_dense:
        #     cls_x=[
        #         self.dense_concat,
        #         self.res_dense_block1,
        #         self.res_dense_block2,
        #         # self.classifier,
        #     ]
        #     for model in cls_x:
        #         # print(model)
        #         for param in model.parameters():
        #             cls_x_regul += torch.norm(param, lp) ** 1
        #             cls_x_param_num += param.numel()

        # if self.extra_classifier:
        #     cls_y=[
        #         self.dense_concat_ab,
        #         self.res_dense_block3,
        #         self.res_dense_block4,
        #         # self.classifier_ar,
        #     ]
        #     for model in cls_y:
        #         for param in model.parameters():
        #             cls_y_regul += torch.norm(param, lp) ** 2
        #             cls_y_param_num += param.numel()
        # return [0.01*cls_x_regul/cls_x_param_num,0.01*cls_y_regul/cls_y_param_num]
        return [0.01 * total_regul / total_param_num,0.01*cls_x_regul/cls_x_param_num]

    def forward(self, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None,ab_bert=None):

        # if ab_bert is not None:
        if True:
            ab_x = ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state

        # print(ag_x)
        # print(ag_edge_index)

        ab_x = torch.permute(ab_x, [0, 2, 1])
        # antibody acnn
        ab_h1 = self.ab_gnn1(ab_x)
        ab_h1 = self.ab_gnn2(ab_h1)

        # predict bind or not
        x2 = torch.nn.functional.max_pool1d(ab_h1, kernel_size=ab_h1.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        # x = torch.cat([x2, x3], 1)  # batch_size * 124
        x = self.dense_concat(x2)
        x = self.classifier(x + self.res_dense_block(x))
        x = self.sigmoid(x)
        x = torch.clamp(x, min=0.001, max=0.999)
        return [x]

class XBCR_ACNN_woBERT_meta(ReptileModel):

    def __init__(self, node_attr_dim=20, hidden_dim=128, h1_dim=64, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=3, freeze_bert=False, ab_freeze_layer_count=None, extra_dense=False):
        super(XBCR_ACNN_woBERT_meta, self).__init__()
        self.freeze_bert=freeze_bert
        self.ab_freeze_layer_count=ab_freeze_layer_count
        # self.ab_bert = BertModel.from_pretrained("train_2_193999")
        # self.ab_bert = BertModel.from_pretrained("prot_bert")
        # freeze the ? layers
        # if ab_freeze_layer_count is not None:
        #     _freeze_bert(self.ab_bert, freeze_bert=freeze_bert, freeze_layer_count=ab_freeze_layer_count)

        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.h1_dim = h1_dim  # dim after the first gnn
        self.h2_dim = h2_dim  # dim after the bipartite
        self.share_weight = share_weight
        self.n_dropout = dropout
        self.heads = heads
        self.block_num = block_num
        self.relu = torch.nn.LeakyReLU()
        # self.gmp = torch.nn.MaxPool1d()
        self.dropout = torch.nn.Dropout(dropout)
        # self.ag_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
        self.ag_gnn1 = Atrous_block(20, self.hidden_dim)
        atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
        self.ag_gnn2 = torch.nn.Sequential(*atrous_blocks)
        self.sigmoid = torch.nn.Sigmoid()
        if self.share_weight:
            self.ab_gnn1 = self.ag_gnn1
            self.ab_gnn2 = self.ag_gnn2
        else:
            self.ab_gnn1 = Atrous_block(self.node_attr_dim, self.hidden_dim)
            atrous_blocks = [Atrous_block(self.hidden_dim, self.hidden_dim) for _ in range(block_num)]
            self.ab_gnn2 = torch.nn.Sequential(*atrous_blocks)

        self.dense_concat = torch.nn.Linear(3 * self.hidden_dim, self.h1_dim)
        self.res_dense_block = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            torch.nn.Linear(self.h1_dim, self.h1_dim),
            self.relu,
            # torch.nn.Dropout(dropout),
            torch.nn.Linear(self.h1_dim, self.h1_dim))
        self.classifier = torch.nn.Sequential(
            self.relu,
            torch.nn.Linear(self.h1_dim, 1))

    def clone_meta(self):
        clone = XBCR_ACNN_woBERT_meta(node_attr_dim=self.node_attr_dim, hidden_dim=self.hidden_dim, h1_dim=self.h1_dim, h2_dim=self.h2_dim, share_weight=self.share_weight, dropout=self.n_dropout,
                 heads=self.heads, block_num=self.block_num, freeze_bert=self.freeze_bert, ab_freeze_layer_count=self.ab_freeze_layer_count)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda()
        return clone

    def get_variables(self,lp=1):
        self.device=next(self.dense_concat.parameters()).device
        total_regul=torch.tensor(0,dtype=torch.float,device=self.device)
        total_param_num = 1
        for param in self.parameters():
            total_regul += torch.norm(param, lp) ** 1
            total_param_num += param.numel()
        cls_x_regul,cls_y_regul=torch.tensor(0,dtype=torch.float,device=self.device),torch.tensor(0,dtype=torch.float,device=self.device)
        cls_x_param_num, cls_y_param_num = 1, 1
        # if self.extra_dense:
        #     cls_x=[
        #         self.dense_concat,
        #         self.res_dense_block1,
        #         self.res_dense_block2,
        #         # self.classifier,
        #     ]
        #     for model in cls_x:
        #         # print(model)
        #         for param in model.parameters():
        #             cls_x_regul += torch.norm(param, lp) ** 1
        #             cls_x_param_num += param.numel()

        # if self.extra_classifier:
        #     cls_y=[
        #         self.dense_concat_ab,
        #         self.res_dense_block3,
        #         self.res_dense_block4,
        #         # self.classifier_ar,
        #     ]
        #     for model in cls_y:
        #         for param in model.parameters():
        #             cls_y_regul += torch.norm(param, lp) ** 2
        #             cls_y_param_num += param.numel()
        # return [0.01*cls_x_regul/cls_x_param_num,0.01*cls_y_regul/cls_y_param_num]
        return [0.01 * total_regul / total_param_num,0.01*cls_x_regul/cls_x_param_num]

    def forward(self, ag_x, ag_edge_index=None, ag_x_batch=None, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ab_edge_index=None, ab_x_batch=None,ab_bert=None):
        # ab_x = self.ab_bert(input_ids=ab_x, attention_mask=attention_mask_ab_v).last_hidden_state
        # ab_l = self.ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
        # antigen acnn
        # print(ag_x)
        # print(ag_edge_index)
        ag_x = torch.permute(ag_x, [0, 2, 1])
        ab_x = torch.permute(ab_x, [0, 2, 1])
        ab_l = torch.permute(ab_l, [0, 2, 1])

        ag_h1 = self.ag_gnn1(ag_x)
        ag_h1 = self.ag_gnn2(ag_h1)

        # antibody acnn
        ab_h1 = self.ab_gnn1(ab_x)
        ab_h1 = self.ab_gnn2(ab_h1)

        ab_l1 = self.ab_gnn1(ab_l)
        ab_l1 = self.ab_gnn2(ab_l1)

        # predict bind or not
        x1 = torch.nn.functional.max_pool1d(ag_h1, kernel_size=ag_h1.size()[2:])[..., 0]
        x2 = torch.nn.functional.max_pool1d(ab_h1, kernel_size=ab_h1.size()[2:])[..., 0]
        x3 = torch.nn.functional.max_pool1d(ab_l1, kernel_size=ab_l1.size()[2:])[..., 0]
        # x = torch.cat([self.dropout(x1), self.dropout(x2), self.dropout(x3)], 1)  # batch_size * 124
        x = torch.cat([x1, x2, x3], 1)  # batch_size * 124
        x = self.dense_concat(x)
        x = self.classifier(x + self.res_dense_block(x))
        x = self.sigmoid(x)

        return [x]

def _freeze_bert(
        bert_model: BertModel, freeze_bert=True, freeze_layer_count=-1
):
    """Freeze parameters in BertModel (in place)
    Args:
        bert_model: HuggingFace bert model
        freeze_bert: Bool whether to freeze the bert model
        freeze_layer_count: If freeze_bert, up to what layer to freeze.
    Returns:
        bert_model
    """
    if freeze_bert:
        # freeze the entire bert model
        for param in bert_model.parameters():
            param.requires_grad = False
    else:
        # freeze the embeddings
        for param in bert_model.embeddings.parameters():
            param.requires_grad = False
        if freeze_layer_count != -1:
            if freeze_layer_count > 0 :
                # freeze layers in bert_model.encoder
                for layer in bert_model.encoder.layer[:freeze_layer_count]:
                    for param in layer.parameters():
                        param.requires_grad = False

            if freeze_layer_count < 0 :
                # freeze layers in bert_model.encoder
                for layer in bert_model.encoder.layer[freeze_layer_count:]:
                    for param in layer.parameters():
                        param.requires_grad = False
    return None

def get_frozen_bert(key_word="prot_bert"):
    ab_bert=BertModel.from_pretrained(key_word)
    _freeze_bert(ab_bert, freeze_bert=True, freeze_layer_count=None)
    return ab_bert

def get_unfrozen_bert(key_word="prot_bert",n=28):
    ab_bert=BertModel.from_pretrained(key_word)
    _freeze_bert(ab_bert, freeze_bert=False, freeze_layer_count=n)
    return ab_bert

if __name__ == '__main__':
    model = OmniglotModel(20)
    x = torch.autograd.Variable(torch.zeros(5, 28*28))
    y = model(x)
    print('x', x.size())
    print('y', y.size())