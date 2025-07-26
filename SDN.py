# Semantic Disentanglement Network (SDN)模块
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utility import EntropyLoss, MutualInformationEstimator, mutual_information_loss


class SemanticDisentanglementNetwork(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_domains):
        super(SemanticDisentanglementNetwork, self).__init__()
        self.num_domains = num_domains

        # 领域特定类无关特征提取器 (D_s)，每个域一个
        self.domain_specific_disentanglers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ) for _ in range(num_domains)
        ])

        # 互信息估计器，每个域一个
        self.mi_estimators = nn.ModuleList([
            MutualInformationEstimator(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_domains)
        ])

        self.entropy_loss_fn = EntropyLoss() # 重命名以区分方法和实例

    def forward(self, features, di_features, mlp_classifiers, domain_idx=None):
        # 如果提供了domain_idx，只处理特定域
        if domain_idx is not None:
            # 提取域特定类无关特征
            ds_features = self.domain_specific_disentanglers[domain_idx](features)
            # MLP分类器预测
            mlp_output = mlp_classifiers[domain_idx](ds_features)
            # 计算互信息
            mi_loss = mutual_information_loss(self.mi_estimators[domain_idx], di_features, ds_features)

            return {
                'ds_features': ds_features,
                'mlp_output': mlp_output,
                'mi_loss': mi_loss
            }

        # 处理所有域
        all_ds_features = []
        all_mlp_outputs = []
        all_mi_losses = []

        for d in range(self.num_domains):
            # 提取：域特定、类无关特征
            ds_features = self.domain_specific_disentanglers[d](features)
            all_ds_features.append(ds_features)

            # MLP分类器预测
            mlp_output = mlp_classifiers[d](ds_features)
            all_mlp_outputs.append(mlp_output)

            # 计算互信息
            mi_loss = mutual_information_loss(self.mi_estimators[d], di_features, ds_features)
            all_mi_losses.append(mi_loss)

        return {
            'ds_features': all_ds_features,
            'mlp_outputs': all_mlp_outputs,
            'mi_losses': all_mi_losses
        }

    def compute_entropy_loss(self, features, mlp_classifiers, domains):
        ent_loss = 0
        num_valid_domains = 0
        for d in range(self.num_domains):
            domain_mask = (domains == d)
            if domain_mask.sum() > 0:
                # 从 f_g 提取当前域的 f_si
                domain_features_g = features[domain_mask]
                ds_features = self.domain_specific_disentanglers[d](domain_features_g)

                # 使用对应的 MLP 计算输出
                mlp_output = mlp_classifiers[d](ds_features)

                # 计算熵损失
                ent_loss += self.entropy_loss_fn(mlp_output)
                num_valid_domains += 1

        return ent_loss / num_valid_domains if num_valid_domains > 0 else torch.tensor(0.0, device=features.device)