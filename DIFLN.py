# Domain-invariant Features Learning Network
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utility import ConsistencyRegLoss, weak_augmentation, strong_augmentation # 假设 augmentation 在 Utility 中


class DomainInvariantFeaturesLearningNetwork(nn.Module):
    def __init__(self, feature_dim, num_classes, num_domains, hidden_dim=256):
        super(DomainInvariantFeaturesLearningNetwork, self).__init__()
        self.num_domains = num_domains
        self.num_classes = num_classes

        # 语义特征分离器
        self.semantic_disentangler = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # MLP分类器，每个域一个
        self.mlp_classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_domains)
        ])

        # GCN分类器组件
        self.edge_network = EdgeNetwork(hidden_dim)
        self.node_classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_domains)
        ])

        self.consistency_loss = ConsistencyRegLoss()

    def forward(self, features, domain_idx=None):
        # 获取域不变特征
        di_features = self.semantic_disentangler(features)

        # MLP分类器预测
        mlp_outputs = [classifier(di_features) for classifier in self.mlp_classifiers]

        output_dict = {
            'di_features': di_features,
            'mlp_outputs': mlp_outputs
        }
        if domain_idx is not None:
            output_dict['mlp_output'] = mlp_outputs[domain_idx]

        return output_dict

    def compute_gcn_loss(self, di_features, labels, domains):
        # 构建图
        edge_weights = self.edge_network(di_features, labels)

        # 更新节点特征
        updated_features = self.update_node_features(di_features, edge_weights)

        # 计算GCN分类损失
        gcn_loss = 0
        for d in range(self.num_domains):
            domain_mask = (domains == d)
            if domain_mask.sum() > 0:
                domain_features = updated_features[domain_mask]
                domain_labels = labels[domain_mask]
                domain_output = self.node_classifiers[d](domain_features)
                gcn_loss += F.cross_entropy(domain_output, domain_labels)

        gcn_loss /= self.num_domains
        return gcn_loss, edge_weights

    def update_node_features(self, node_features, edge_weights):
        # 简化的GCN节点特征更新
        batch_size = node_features.size(0)
        updated_features = node_features.clone()

        for i in range(batch_size):
            weighted_sum = torch.zeros_like(node_features[0])
            weight_sum = 0

            for j in range(batch_size):
                if i != j:
                    weight = edge_weights[i, j]
                    weighted_sum += weight * node_features[j]
                    weight_sum += weight

            if weight_sum > 0:
                updated_features[i] = updated_features[i] + weighted_sum / weight_sum

        return updated_features

    def compute_cr_loss(self, features, x_raw, domains):
        batch_size = x_raw.size(0)
        cr_loss = 0

        for d in range(self.num_domains):
            domain_mask = (domains == d)
            if domain_mask.sum() > 0:
                domain_samples_raw = x_raw[domain_mask]

                # 增强版本 - 注意：需要原始图像数据进行增强
                weak_aug_samples = weak_augmentation(domain_samples_raw)
                strong_aug_samples = strong_augmentation(domain_samples_raw)

                # !!! 重要: 需要重新提取增强后样本的特征
                # !!! 这意味着 feature_extractor 还是需要被访问
                # !!! 方案1: 将 feature_extractor 重新传给 DIFLN
                # !!! 方案2: 在 D2IFLN 中计算 CR Loss，因为那里有 feature_extractor
                # 暂时假设 feature_extractor 仍然可用 (需要调整 D2IFLN 初始化)
                # 或者更好的方式是在 D2IFLN.compute_losses 中计算 cr_loss

                # 假设在 D2IFLN 中计算，这里暂时注释掉
                pass # CR Loss 逻辑移至 D2IFLN.compute_losses

        return torch.tensor(0.0, device=features.device) # 占位符，实际计算移到 D2IFLN


class EdgeNetwork(nn.Module):
    def __init__(self, feature_dim):
        super(EdgeNetwork, self).__init__()
        self.edge_network = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, features, labels=None):
        # 计算特征之间的边权重
        batch_size = features.size(0)
        edge_weights = torch.zeros(batch_size, batch_size, device=features.device)

        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    # 如果提供了标签，只在同类样本之间建立边
                    if labels is not None and labels[i] != labels[j]:
                        continue

                    # 计算特征对的相似度
                    pair_features = torch.cat([features[i], features[j]], dim=0)
                    edge_weights[i, j] = self.edge_network(pair_features.unsqueeze(0)).squeeze()

        return edge_weights