# Domain Disentanglement Network (DDN)模块
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utility import MutualInformationEstimator, mutual_information_loss, grad_reverse


class DomainDisentanglementNetwork(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_domains):
        super(DomainDisentanglementNetwork, self).__init__()

        # 域判别特征提取器
        self.domain_disentangler = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # 域分类器
        self.domain_classifier = nn.Linear(hidden_dim, num_domains)

        # 互信息估计器
        self.mi_estimator = MutualInformationEstimator(hidden_dim, hidden_dim, hidden_dim)

    def forward(self, features, di_features=None, alpha=1.0):
        # 提取域判别特征
        dd_features = self.domain_disentangler(features)

        # 域分类
        domain_output = self.domain_classifier(dd_features)

        # 如果提供了di_features，通过梯度反转后进行域分类
        di_domain_output = None
        if di_features is not None:
            # 应用梯度反转
            reversed_di_features = grad_reverse(di_features, alpha)
            # 域分类
            di_domain_output = self.domain_classifier(reversed_di_features)

        # 计算互信息
        mi_loss = None
        if di_features is not None:
            mi_loss = mutual_information_loss(self.mi_estimator, di_features, dd_features)

        return {
            'dd_features': dd_features,
            'domain_output': domain_output,
            'di_domain_output': di_domain_output,
            'mi_loss': mi_loss
        }