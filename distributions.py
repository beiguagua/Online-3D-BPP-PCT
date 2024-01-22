import torch
#
# """
# Modify standard PyTorch distributions so they are compaible with this code.
# """
#
# Categorical
# FixedCategorical = torch.distributions.Categorical
#
# old_sample = FixedCategorical.sample
# FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)
#
# log_prob_cat = FixedCategorical.log_prob
# FixedCategorical.log_probs = lambda self, actions: log_prob_cat(
#     self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)
#
# FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)
#
# # Normal
# FixedNormal = torch.distributions.Normal
#
# log_prob_normal = FixedNormal.log_prob
# FixedNormal.log_probs = lambda self, actions: log_prob_normal(
#     self, actions).sum(
#         -1, keepdim=True)
#
# normal_entropy = FixedNormal.entropy
# FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)
#
# FixedNormal.mode = lambda self: self.mean
#
# # Bernoulli
# FixedBernoulli = torch.distributions.Bernoulli
#
# log_prob_bernoulli = FixedBernoulli.log_prob
# FixedBernoulli.log_probs = lambda self, actions: log_prob_bernoulli(
#     self, actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)
#
# bernoulli_entropy = FixedBernoulli.entropy
# FixedBernoulli.entropy = lambda self: bernoulli_entropy(self).sum(-1)
# FixedBernoulli.mode = lambda self: torch.gt(self.probs, 0.5).float()



import torch
import torch.jit

# @torch.jit.script
def calculate_entropy(probs):
    # 计算熵
    return -torch.sum(probs * torch.log(probs), dim=-1)

# @torch.jit.script
class FixedCategoricalScript:
    def __init__(self, probs):
        self.probs = probs

    def sample(self):
        return torch.multinomial(self.probs, 1)

    def log_probs(self, actions):
        return torch.log(self.probs.gather(-1, actions))

    def entropy(self):
        # 使用手动计算的熵
        return calculate_entropy(self.probs)

    def mode(self):
        _, mode = torch.max(self.probs, dim=-1, keepdim=True)
        return mode

# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean

# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super().log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()
