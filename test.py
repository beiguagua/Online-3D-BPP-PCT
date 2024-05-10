# import numpy as np
#
# # 打开.npy文件
# data = np.load('./logs/evaluation/BP-Mask-600-60-80-rs1-2024.03.21-09-13-57/trajs.npy')
#
# # 现在你可以使用`data`变量来访问文件中的数组了
# print(data)

import torch
from tools import get_args,load_policy

from model import DRL_GAT

args = get_args()
args.setting = 1
args.internal_node_holder=60
args.leaf_node_holder=80
args.next_holder=1
args.lnes='BP'

agent = DRL_GAT(args)

PCT_policy = load_policy('logs/experiment/test-2024.04.25-10-45-50/PCT-test-2024.04.25-10-45-50_2024.05.02-09-08-14.pt', agent)
PCT_policy.eval()
scripted_model = torch.jit.script(agent)
scripted_model.save('BP_Mask_10_60_80.pt')