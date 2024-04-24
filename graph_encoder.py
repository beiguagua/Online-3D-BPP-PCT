import torch
from torch import nn
import math
from typing import Tuple
from tools import clones


# class SkipConnection(nn.Module):
#     def __init__(self, module):
#         super(SkipConnection, self).__init__()
#         self.module = module
#
#     def forward(self, input):
#         return {'data': input['data'] + self.module(input), 'mask': input['mask'], 'graph_size': input['graph_size'],
#                 'evaluate': input['evaluate']}
#
#
# class SkipConnection_Linear(nn.Module):
#     def __init__(self, module):
#         super(SkipConnection_Linear, self).__init__()
#         self.module = module
#
#     def forward(self, input):
#         return {'data': input['data'] + self.module(input['data']), 'mask': input['mask'],
#                 'graph_size': input['graph_size'], 'evaluate': input['evaluate']}


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads: int,
            input_dim: int,
            embed_dim: int = 0,
            val_dim: int = 0,
            key_dim: int = 0,
            internal_node_holder: int = 0,
            leaf_node_holder: int = 0,
            next_holder: int = 0
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is 0:
            assert embed_dim is not 0, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is 0:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.internal_node_holder = internal_node_holder
        self.leaf_node_holder = leaf_node_holder
        self.next_holder = next_holder

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Linear(input_dim, key_dim, bias=False)
        self.W_key = nn.Linear(input_dim, key_dim, bias=False)
        self.W_val = nn.Linear(input_dim, val_dim, bias=False)

        if embed_dim is not 0:
            # self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))
            self.W_out = nn.Linear(key_dim, embed_dim)

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q: torch.Tensor, mask: torch.Tensor, graph_size: int, evaluate: bool,
                h: torch.Tensor) -> torch.Tensor:
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :param graph_size: Size of the graph.
        :param evaluate: Whether to perform evaluation.
        :return:
        """

        batch_size = int(q.size()[0] / graph_size)
        graph_size = graph_size
        input_dim = h.size()[-1]
        n_query = graph_size
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)
        Q = self.W_query(qflat).view(shp_q)
        K = self.W_key(hflat).view(shp)
        V = self.W_val(hflat).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        mask = mask.unsqueeze(1).repeat(*(1, graph_size, 1)) > 0
        mask[:, self.internal_node_holder:-self.next_holder, self.internal_node_holder:-self.next_holder] = True
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            if evaluate:
                compatibility[mask] = -math.inf
            else:
                compatibility[mask] = -30
        attn = torch.softmax(compatibility, dim=-1)  #

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)
        out = self.W_out(heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim)).view(
            batch_size * n_query, self.embed_dim)
        return out


class MultiHeadAttentionLayer(nn.Module):
    def __init__(
            self,
            n_heads: int,
            embed_dim: int,
            feed_forward_hidden: int = 128,
            internal_node_holder: int = 0,
            leaf_node_holder: int = 0,
            next_holder: int = 0):
        super(MultiHeadAttentionLayer, self).__init__()
        self.mha = MultiHeadAttention(
            n_heads,
            input_dim=embed_dim,
            embed_dim=embed_dim,
            internal_node_holder=internal_node_holder,
            leaf_node_holder=leaf_node_holder,
            next_holder=next_holder
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden, embed_dim)
        ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)

    def forward(self, data: torch.Tensor, mask: torch.Tensor, graph_size: int, evaluate: bool) -> torch.Tensor:
        data = data + self.mha(data, mask=mask, graph_size=graph_size, evaluate=evaluate, h=data)
        data = data + self.mlp(data)
        return data


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads: int,
            embed_dim: int,
            n_layers: int,
            node_dim: int = 0,
            feed_forward_hidden: int = 128,
            internal_node_holder: int = 0,
            leaf_node_holder: int = 0,
            next_holder: int = 0
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim != 0 else None
        self.graph_size = internal_node_holder+leaf_node_holder+next_holder
        mha_layer=MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden,
                                              internal_node_holder=internal_node_holder,
                                              leaf_node_holder=leaf_node_holder, next_holder=next_holder)
        self.layers = clones(mha_layer,n_layers)
        # self.layers = MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden,
        #                                       internal_node_holder=internal_node_holder,
        #                                       leaf_node_holder=leaf_node_holder, next_holder=next_holder)

        # self.layers = nn.Sequential(*(
        #     MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden,internal_node_holder=internal_node_holder,)
        #     for _ in range(n_layers)
        # ))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, evaluate: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        for layer in self.layers:
            h=layer(h, mask=mask, graph_size=self.graph_size, evaluate=evaluate)

        # h = self.layers(h, mask=mask, graph_size=self.graph_size, evaluate=evaluate)
        # data = {'data':h, 'mask': mask, 'graph_size': self.graph_size, 'evaluate': evaluate}
        # h = self.layers(data)['data']
        return (h, h.view(int(h.size()[0] / self.graph_size), self.graph_size, -1).mean(dim=1),)
