import torch
import random

from givenData import *


# import numpy as np
# data = np.loadtxt("./datasets/itemset/RS50.txt")
# items = []
# for _ in range(200):
#     items.append(data)
# data_tensor=torch.tensor(items)
# torch.save(data_tensor, './datasets/itemset/rs50.pt')

# 保存每轮选择的数据的列表
selected_items = []

# 进行2000轮选择
for _ in range(2000):
    # 每轮选择200次
    selected_round = []
    for _ in range(200):
        # 随机选择一个元素并添加到选定的数据列表中
        selected_round.append(random.choice(item_size_set))
    selected_items.append(selected_round)

# 将选定的数据转换为PyTorch张量
data_tensor = torch.tensor(selected_items)

# 保存为.pt文件
torch.save(data_tensor, './datasets/itemset/rs.pt')