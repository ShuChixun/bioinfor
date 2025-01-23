import random

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import add_self_loops, negative_sampling

num_nodes = 8  # 节点数n=2655
num_classes = 3  # 5种节点
num_edges = 12  # 单向边的数量

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 确保 num_nodes // num_classes 至少为 1
min_nodes_per_class = max(1, num_nodes // num_classes)
# 生成5个随机数来确定每种节点的数量
node_counts = torch.randint(min_nodes_per_class, num_nodes, (num_classes,))
node_counts = node_counts.float() / node_counts.sum() * num_nodes
node_counts = node_counts.int()
# 确保节点总数为num_nodes
node_counts[-1] = num_nodes - node_counts[:-1].sum()

# 累加计算每种节点的起始索引
start_indices = torch.cumsum(torch.cat((torch.tensor([0]), node_counts[:-1])), dim=0)

# 生成节点类别
y = torch.cat([torch.full((count,), i, dtype=torch.long) for i, count in enumerate(node_counts)])

x = torch.randn(num_nodes, 128)  # 随机初始化节点特征
edge_attr = torch.ones(num_edges*2, 1, dtype=torch.float)  # 假设全是正边，边的标签全为1

# 生成所有可能的节点对
class_pairs = torch.combinations(torch.arange(num_classes), 2)

# 计算每种边的数量
num_edge_types = len(class_pairs)
min_edges_per_type = max(1, num_edges // num_edge_types)
edge_counts = torch.randint(min_edges_per_type, num_edges, (num_edge_types,))
edge_counts = edge_counts.float() / edge_counts.sum() * num_edges
edge_counts = edge_counts.int()
# 确保边总数不超过num_edges
edge_counts[-1] = num_edges - edge_counts[:-1].sum()
edge_counts = edge_counts * 2

# 根据节点类别生成边和边类型
edge_index = []
edge_type = []

for pair_id, (class_id1, class_id2) in enumerate(class_pairs):
    # 获取当前类别的节点索引
    class_nodes1 = start_indices[class_id1] + torch.arange(node_counts[class_id1])
    class_nodes2 = start_indices[class_id2] + torch.arange(node_counts[class_id2])
    num_class_nodes1 = class_nodes1.size(0)
    num_class_nodes2 = class_nodes2.size(0)

    # 生成边索引
    if num_class_nodes1 > 0 and num_class_nodes2 > 0:
        num_edges = min(num_class_nodes1 * num_class_nodes2, edge_counts[pair_id] // 2)
        edges = torch.cartesian_prod(class_nodes1, class_nodes2)
        selected_edges = edges[torch.randperm(edges.size(0))[:num_edges]]

        # 添加正向边
        edge_index.append(selected_edges.t())
        edge_type.append(torch.full((num_edges,), pair_id, dtype=torch.long))

        # 添加反向边
        edge_index.append(selected_edges.flip(1).t())
        edge_type.append(torch.full((num_edges,), pair_id, dtype=torch.long))

# 合并所有边索引和边类型
edge_index = torch.cat(edge_index, dim=1)
edge_type = torch.cat(edge_type)

# 创建图数据集
data = Data(x=x, y=y, edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr)

# save data with torch.save function which has faster processing.
# torch.save(data, './data/graph.pt')
# data = torch.load('./test.pt', weights_only=False)
edge_index = data.edge_index
valid_ratio = 0.1
test_ratio = 0.1
transform = RandomLinkSplit(num_val=valid_ratio, num_test=test_ratio, is_undirected=True, split_labels=True)
train_data, val_data, test_data = transform(data)
split_edge = {'biased_train': {}, 'valid': {}, 'test': {}, 'train': {}}

edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)

train_pos_edge_index = torch.cat(
    (train_data.pos_edge_label_index, torch.flip(train_data.pos_edge_label_index, dims=[0])), dim=1)
# 负采样完全是随机的
train_neg_edge_index = negative_sampling(edge_index, num_nodes=train_data.num_nodes,
                                         num_neg_samples=train_data.edge_index.size(1))

split_edge['biased_train']['edge'] = train_pos_edge_index.t()
split_edge['biased_train']['edge_neg'] = train_neg_edge_index.t()

split_edge['train']['edge'] = train_pos_edge_index.t()
train_edge_neg_mask = torch.ones((train_data.num_nodes, train_data.num_nodes), dtype=torch.bool)
train_edge_neg_mask[tuple(split_edge['train']['edge'].t().tolist())] = False
train_edge_neg_mask = torch.triu(train_edge_neg_mask, 1)
split_edge['train']['edge_neg'] = torch.nonzero(train_edge_neg_mask)

val_pos_edge_index = torch.cat(
    (val_data.pos_edge_label_index, torch.flip(val_data.pos_edge_label_index, dims=[0])), dim=1)
split_edge['valid']['edge'] = val_pos_edge_index.t()
valid_edge_neg_mask = train_edge_neg_mask.clone()
valid_edge_neg_mask[tuple(split_edge['valid']['edge'].t().tolist())] = False
split_edge['valid']['edge_neg'] = torch.nonzero(valid_edge_neg_mask)

test_pos_edge_index = torch.cat(
    (test_data.pos_edge_label_index, torch.flip(test_data.pos_edge_label_index, dims=[0])), dim=1)
split_edge['test']['edge'] = test_pos_edge_index.t()
test_edge_neg_mask = valid_edge_neg_mask.clone()
test_edge_neg_mask[tuple(split_edge['test']['edge'].t().tolist())] = False
split_edge['test']['edge_neg'] = torch.nonzero(test_edge_neg_mask)

print(split_edge)
