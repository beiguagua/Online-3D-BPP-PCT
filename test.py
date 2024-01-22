import torch
from tools import get_args,load_policy

from model import DRL_GAT

args = get_args()
args.setting = 1

agent = DRL_GAT(args)

PCT_policy = load_policy('models/PCT-origin-2024.01.20-15-11-12_2024.01.22-04-38-30.pt', agent)
PCT_policy.eval()
scripted_model = torch.jit.script(agent)
scripted_model.save('model.pt')
