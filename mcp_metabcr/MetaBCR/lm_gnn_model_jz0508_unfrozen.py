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
        self.evolut_fusion = torch.nn.Sequential(
            torch.nn.Conv2d(self.num_heads * 2, self.num_heads * 4, kernel_size=3, padding=1, stride=1),
            torch.nn.LeakyReLU(negative_slope=1e-1, inplace=True),
            torch.nn.Conv2d(self.num_heads * 4, self.num_heads, kernel_size=3, padding=1, stride=1))

        self.evolut_fusion = torch.nn.Conv2d(self.num_heads * 2, self.num_heads, kernel_size=3, padding=1, stride=1)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            pos_embedding: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            edge_weights: Optional[torch.Tensor] = None,
    ) -> (torch.Tensor, torch.Tensor):
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
            score = self.evolut_fusion(torch.cat([score, edge_weights], 1))
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


class Adaptive_Regulariz():
    def __init__(self, dropout_range=[0., 0.9], weight_range=[0., 10.], velocity=[0.02, 0.001],
                 target_deviation_ratio=0.3):
        self.dropout = 0.5
        # self.weight=0.1
        self.weight = 0.0
        self.dropout_range = dropout_range
        self.weight_range = weight_range
        self.dropout_velocity = velocity[0]
        self.weight_velocity = velocity[1]
        self.thresh = target_deviation_ratio
        return

    def _update_scale(self, loss_src, loss_tgt, eps=10e-8):
        return (loss_tgt - loss_src) / (abs(loss_tgt) + eps) - self.thresh
        # return np.maximum((loss_tgt-loss_src)/abs(loss_tgt)-self.thresh,0)
        # return (abs(loss_tgt)-abs(loss_src))/abs(loss_tgt)-self.thresh

    def update_dropout(self, loss_src, loss_tgt):
        scale = self._update_scale(loss_src, loss_tgt)
        self.dropout += self.dropout_velocity * scale
        return np.clip(self.dropout, self.dropout_range[0], self.dropout_range[1]).item()

    def update_weight(self, loss_src, loss_tgt):
        scale = self._update_scale(loss_src, loss_tgt)
        self.weight += self.weight_velocity * scale
        # return np.clip(self.weight, self.weight_range[0], self.weight_range[1]).item()
        self.weight=np.clip(self.weight, self.weight_range[0], self.weight_range[1]).item()
        return self.weight

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


class ReptileModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def point_grad_to(self, target, device=torch.device('cuda')):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if self.is_cuda():
                    p.grad = torch.autograd.Variable(torch.zeros(p.size())).to(device)
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


class XBCR_ACNN_dense_meta(ReptileModel):

    def __init__(self, node_attr_dim=1024, hidden_dim=128, h1_dim=256, h2_dim=16, share_weight=False, dropout=0.,
                 heads=4, block_num=2, freeze_bert=False, ab_freeze_layer_count=None, extra_dense=True,
                 bert=None):
        super(ReptileModel, self).__init__()
        expand_num = 4
        # self.ab_bert = BertModel.from_pretrained("train_2_193999")
        self.bert_name = bert
        self.ab_bert = BertModel.from_pretrained(self.bert_name)
        # freeze the ? layers
        self.freeze_bert = freeze_bert
        self.ab_freeze_layer_count = ab_freeze_layer_count

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
        self.extra_dense = extra_dense
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

    def clone_meta(self, device=torch.device('cuda')):
        clone = XBCR_ACNN_dense_meta(node_attr_dim=self.node_attr_dim, hidden_dim=self.hidden_dim, h1_dim=self.h1_dim,
                                     h2_dim=self.h2_dim, share_weight=self.share_weight, dropout=self.n_dropout,
                                     heads=self.heads, block_num=self.block_num, freeze_bert=self.freeze_bert,
                                     ab_freeze_layer_count=self.ab_freeze_layer_count, bert=self.bert_name)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.to(device)
        return clone

    def get_variables(self, lp=1):
        self.device = next(self.dense_concat.parameters()).device
        total_regul = torch.tensor(0, dtype=torch.float, device=self.device)
        total_param_num = 1
        for param in self.parameters():
            total_regul += torch.norm(param, lp) ** 1
            total_param_num += param.numel()
        cls_x_regul, cls_y_regul = torch.tensor(0, dtype=torch.float, device=self.device), torch.tensor(0,
                                                                                                        dtype=torch.float,
                                                                                                        device=self.device)
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
        return [0.01 * total_regul / total_param_num, 0.01 * cls_x_regul / cls_x_param_num]

    def forward(self, ag_x, ab_x=None, attention_mask_ab_v=None, ab_l=None,
                attention_mask_ab_l=None, ):

        # if ab_bert is not None:
        if True:
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
            if freeze_layer_count > 0:
                # freeze layers in bert_model.encoder
                for layer in bert_model.encoder.layer[:freeze_layer_count]:
                    for param in layer.parameters():
                        param.requires_grad = False

            if freeze_layer_count < 0:
                # freeze layers in bert_model.encoder
                for layer in bert_model.encoder.layer[freeze_layer_count:]:
                    for param in layer.parameters():
                        param.requires_grad = False
    return None


if __name__ == '__main__':
    model = OmniglotModel(20)
    x = torch.autograd.Variable(torch.zeros(5, 28 * 28))
    y = model(x)
    print('x', x.size())
    print('y', y.size())
