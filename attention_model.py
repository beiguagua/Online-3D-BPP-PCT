import torch
from torch import nn
import math
from typing import NamedTuple, Tuple, Optional
from graph_encoder import GraphAttentionEncoder
from distributions import FixedCategoricalScript
from tools import observation_decode_leaf_node, observation_to_node
from position_encoding import VolumetricPositionEncoding


class AttentionModelFixed(NamedTuple):
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    # def __getitem__(self, key):
    #     if torch.is_tensor(key) or isinstance(key, slice):
    #         return AttentionModelFixed(
    #             node_embeddings=self.node_embeddings[key],
    #             context_node_projected=self.context_node_projected[key],
    #             glimpse_key=self.glimpse_key[:, key],
    #             glimpse_val=self.glimpse_val[:, key],
    #             logit_key=self.logit_key[key]
    #         )
    #     return super(AttentionModelFixed, self).__getitem__(key)


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 n_encode_layers: int = 2,
                 tanh_clipping: int = 10.,
                 mask_inner: bool = False,
                 mask_logits: bool = False,
                 n_heads: int = 1,
                 internal_node_holder: int = 80,
                 internal_node_length: int = 6,
                 leaf_node_holder: int = 50,
                 next_holder: int = 1
                 ):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.temp: float = 1.0

        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.n_heads = n_heads
        self.internal_node_holder = internal_node_holder
        self.internal_node_length = internal_node_length
        self.next_holder = next_holder
        self.leaf_node_holder = leaf_node_holder

        # graph_size = internal_node_holder + leaf_node_holder + self.next_holder
        graph_size = internal_node_holder + leaf_node_holder

        activate, ini = nn.LeakyReLU, 'leaky_relu'

        # 定义初始化函数
        def custom_init(m):
            if isinstance(m, nn.Linear):
                # 对 Linear 层使用 xavier_normal_ 进行初始化
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain(ini))
                # 对 bias 使用常数初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

        self.init_internal_node_embed = nn.Sequential(
            nn.Linear(self.internal_node_length, 32),
            activate(),
            nn.Linear(32, embedding_dim))

        self.init_internal_node_embed.apply(custom_init)

        self.init_leaf_node_embed = nn.Sequential(
            nn.Linear(8, 32),
            activate(),
            nn.Linear(32, embedding_dim))
        self.init_leaf_node_embed.apply(custom_init)

        # self.init_next_embed = nn.Sequential(
        #     nn.Linear(6, 32),
        #     activate(),
        #     nn.Linear(32, embedding_dim))
        # self.init_next_embed.apply(custom_init)

        vol_bnds = [[-3.6, -2.4, 1.14], [1.093, 0.78, 2.92]]
        voxel_size = 0.08
        pe_type = "rotary"  # options: [ 'rotary', 'sinusoidal']
        self.position_embed = VolumetricPositionEncoding(embedding_dim, vol_bnds, voxel_size, pe_type)

        # Graph attention model
        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            internal_node_holder=self.internal_node_holder,
            leaf_node_holder=self.leaf_node_holder,
            next_holder=self.next_holder
        )

        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0

    def forward(self, input: torch.Tensor, deterministic: bool = False, evaluate_action: bool = False,
                normFactor: float = 1, evaluate: bool = False) -> Tuple[
        Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, FixedCategoricalScript]:

        # internal_nodes, leaf_nodes, next_item, invalid_leaf_nodes, full_mask = observation_decode_leaf_node(input,
        #                                                                                                     self.internal_node_holder,
        #                                                                                                     self.internal_node_length,
        #                                                                                                     self.leaf_node_holder)
        position, internal_nodes, leaf_nodes, next_item, invalid_leaf_nodes, full_mask = observation_to_node(input,
                                                                                                             self.internal_node_holder,
                                                                                                             self.internal_node_length,
                                                                                                             self.leaf_node_holder)
        leaf_node_mask = 1 - invalid_leaf_nodes   # 0:valid,1:invalid
        # has_negative = torch.any(leaf_node_mask > 1)
        valid_length: torch.Tensor = full_mask.sum(1)
        full_mask = 1 - full_mask

        # full_mask = full_mask[:,:-1]

        batch_size = input.size(0)
        # graph_size = input.size(1)
        internal_nodes_size = internal_nodes.size(1)
        leaf_node_size = leaf_nodes.size(1)
        next_size = next_item.size(1)
        graph_size = internal_nodes_size+leaf_node_size

        internal_inputs = internal_nodes.contiguous().view(batch_size * internal_nodes_size,
                                                           self.internal_node_length) * normFactor
        leaf_inputs = leaf_nodes.contiguous().view(batch_size * leaf_node_size, 8) * normFactor
        current_inputs = next_item.contiguous().view(batch_size * next_size, 6) * normFactor

        # We use three independent node-wise Multi-Layer Perceptron (MLP) blocks to project these raw space configuration nodes
        # presented by descriptors in different formats into the homogeneous node features.
        internal_embedded_inputs = self.init_internal_node_embed(internal_inputs).reshape(
            (batch_size, -1, self.embedding_dim))
        leaf_embedded_inputs = self.init_leaf_node_embed(leaf_inputs).reshape((batch_size, -1, self.embedding_dim))
        # next_embedded_inputs = self.init_next_embed(current_inputs.squeeze()).reshape(batch_size, -1,
        #                                                                               self.embedding_dim)
        # init_embedding = torch.cat((internal_embedded_inputs, leaf_embedded_inputs, next_embedded_inputs), dim=1).view(
        #     batch_size * graph_size, self.embedding_dim)

        pe=self.position_embed(position)

        init_embedding = torch.cat((internal_embedded_inputs, leaf_embedded_inputs), dim=1).view(
            batch_size , graph_size, self.embedding_dim)
        
        init_embedding=VolumetricPositionEncoding.embed_pos('rotary',init_embedding,pe)
        
        init_embedding=init_embedding.view(batch_size * graph_size, self.embedding_dim)

        # transform init_embedding into high-level node features.
        embeddings, _ = self.embedder(init_embedding, mask=full_mask, evaluate=evaluate)
        embedding_shape = (batch_size, graph_size, embeddings.shape[-1])

        # Decide the leaf node indices for accommodating the current item
        log_p, action_log_prob, pointers, dist_entropy, dist, hidden = self._inner(embeddings,
                                                                                   deterministic=deterministic,
                                                                                   evaluate_action=evaluate_action,
                                                                                   shape=embedding_shape,
                                                                                   mask=leaf_node_mask,
                                                                                   full_mask=full_mask,
                                                                                   valid_length=valid_length)
        return action_log_prob, pointers, dist_entropy, hidden, dist

    def _inner(self, embeddings: torch.Tensor, mask: torch.Tensor, shape: Tuple[int, int, int], full_mask: torch.Tensor,
               valid_length: torch.Tensor, deterministic: bool = False, evaluate_action: bool = False) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, FixedCategoricalScript, torch.Tensor]:  # 元素齐了
        # The aggregation of global feature
        fixed = self._precompute(embeddings, shape=shape, full_mask=full_mask, valid_length=valid_length)
        # Calculate probabilities of selecting leaf nodes
        log_p, mask = self._get_log_p(fixed, mask)
        # has_negative = torch.any(log_p < 0)
        # The leaf node which is not feasible will be masked in a soft way.
        # has_negative = torch.any(mask > 1)
        if deterministic:
            masked_outs = log_p * (1 - mask)
            if torch.sum(masked_outs) == 0:
                masked_outs += 1e-20
        else:
            masked_outs = log_p * (1 - mask) + 1e-20
        # has_negative = torch.any(masked_outs < 0)
        log_p = torch.div(masked_outs, torch.sum(masked_outs, dim=1).unsqueeze(1))
        # 判断是否存在负数
        # has_negative = torch.any(log_p < 0)
        dist = FixedCategoricalScript(probs=log_p)
        dist_entropy = dist.entropy()

        # Get maximum probabilities and indices
        if deterministic:
            # We take the argmax of the policy for the test.
            selected = dist.mode()
        else:
            # The action at is sampled from the distribution for training
            selected = dist.sample()

        if not evaluate_action:
            action_log_probs = dist.log_probs(selected)
        else:
            action_log_probs = None

        # Collected lists, return Tensor
        return log_p, action_log_probs, selected, dist_entropy, dist, fixed.context_node_projected

    def _precompute(self, embeddings: torch.Tensor, full_mask: torch.Tensor, shape: Tuple[int, int, int],
                    valid_length: torch.Tensor,
                    num_steps: int = 1) -> AttentionModelFixed:
        # The aggregation of global feature, only happens on the eligible nodes.
        transEmbedding = embeddings.view(shape)
        full_mask = full_mask.view(shape[0], shape[1], 1).expand(shape) > 0
        transEmbedding[full_mask] = 0
        graph_embed = transEmbedding.view(shape).sum(1)
        transEmbedding = transEmbedding.view(embeddings.shape)

        graph_embed = graph_embed / valid_length.reshape((-1, 1))
        fixed_context = self.project_fixed_context(graph_embed)
        assert not torch.isnan(fixed_context).any()
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(transEmbedding).view((shape[0], 1, shape[1], -1)).chunk(3, dim=-1)

        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(transEmbedding, fixed_context, *fixed_attention_node_data)

    def _get_log_p(self, fixed: AttentionModelFixed, mask: torch.Tensor, normalize: bool = True) -> Tuple[
        torch.Tensor, torch.Tensor]:
        # Compute query = context node embedding
        query = fixed.context_node_projected[:, None, :]

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed)

        # Compute logits (unnormalized log_p)
        log_p = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        
        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)
        assert not torch.isnan(log_p).any()
        return log_p.exp(), mask

    def _one_to_many_logits(self, query: torch.Tensor, glimpse_K: torch.Tensor, glimpse_V: torch.Tensor,
                            logit_K: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads
        
        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        
        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        
        logits = compatibility.reshape([-1, 1, compatibility.shape[-1]])

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        logits = logits[:, 0, self.internal_node_holder: self.internal_node_holder + self.leaf_node_holder]
        if self.mask_logits:
            logits[mask > 0] = -math.inf

        return logits

    def _get_attention_node_data(self, fixed: AttentionModelFixed) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v: torch.Tensor, num_steps: int) -> torch.Tensor:
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)
        )
