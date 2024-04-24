import torch.nn as nn
import torch
import torch.nn as nn
from numpy import sqrt
from attention_model import AttentionModel
from argparse import Namespace


class DRL_GAT(nn.Module):
    def __init__(self, args: Namespace):
        super(DRL_GAT, self).__init__()

        self.actor = AttentionModel(args.embedding_size,
                                    args.hidden_size,
                                    n_encode_layers=args.gat_layer_num,
                                    n_heads=1,
                                    internal_node_holder=args.internal_node_holder,
                                    internal_node_length=args.internal_node_length,
                                    leaf_node_holder=args.leaf_node_holder,
                                    next_holder=args.next_holder,
                                    )
        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), sqrt(2))
        self.critic = nn.Linear(args.embedding_size, 1)
        nn.init.xavier_normal_(self.critic.weight.data, gain=sqrt(2))

    def forward(self, items: torch.Tensor, deterministic: bool = False, normFactor: float = 1, evaluate: bool = False):
        o, p, dist_entropy, hidden, _ = self.actor(items, deterministic, normFactor=normFactor, evaluate=evaluate)
        values = self.critic(hidden)
        return o, p, dist_entropy, values

    def evaluate_actions(self, items: torch.Tensor, actions: torch.Tensor, normFactor: float = 1):
        _, p, dist_entropy, hidden, dist = self.actor(items, evaluate_action=True, normFactor=normFactor)
        action_log_probs = dist.log_probs(actions)
        values = self.critic(hidden)
        return values, action_log_probs, dist_entropy.mean()
