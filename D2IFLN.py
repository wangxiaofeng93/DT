# 整个D²IFLN架构
import torch
import torch.nn as nn
import torch.nn.functional as F
from DIFLN import DomainInvariantFeaturesLearningNetwork
from SDN import SemanticDisentanglementNetwork
from DDN import DomainDisentanglementNetwork
# 假设 Utility 包含 weak/strong augmentation
from Utility import weak_augmentation, strong_augmentation


class D2IFLN(nn.Module):
    def __init__(self, feature_extractor, feature_dim, num_classes, num_domains, hidden_dim=256):
        super(D2IFLN, self).__init__()

        self.feature_extractor = feature_extractor # 保存共享的特征提取器

        # 初始化 DIFLN (不传入 feature_extractor)
        self.difln = DomainInvariantFeaturesLearningNetwork(
            feature_dim, num_classes, num_domains, hidden_dim
        )

        # 初始化 SDN
        self.sdn = SemanticDisentanglementNetwork(
            feature_dim, hidden_dim, num_domains
        )

        # 初始化 DDN
        self.ddn = DomainDisentanglementNetwork(
            feature_dim, hidden_dim, num_domains
        )

        self.num_domains = num_domains
        self.num_classes = num_classes

    def forward(self, x, domain_idx=None, compute_all=False):
        # 1. 提取共享特征 f_g
        features = self.feature_extractor(x)

        # 2. DIFLN 处理
        difln_output = self.difln(features, domain_idx)
        di_features = difln_output['di_features'] # f_di

        # 3. DDN 处理
        # alpha 控制梯度反转强度，可以设为超参数或随训练变化
        ddn_output = self.ddn(features, di_features, alpha=1.0)

        # 4. SDN 处理
        sdn_output = self.sdn(features, di_features, self.difln.mlp_classifiers, domain_idx)

        # 5. 整合输出 (根据需要调整返回内容)
        outputs = {
            'features': features,       # f_g
            'di_features': di_features,   # f_di
            'dd_features': ddn_output['dd_features'], # f_dd
            'ds_features': sdn_output.get('ds_features'), # f_si (list or tensor)
            'mlp_outputs': difln_output['mlp_outputs'], # 主分类器输出 (list)
            'ddn_domain_output': ddn_output['domain_output'], # F_d(f_dd)
            'di_domain_output': ddn_output['di_domain_output'], # F_d(GRL(f_di)) for adv_loss
            'ddn_mi_loss': ddn_output['mi_loss'], # MI(f_di, f_dd)
            'sdn_mi_losses': sdn_output.get('mi_losses'), # MI(f_di, f_si) (list)
            'sdn_mlp_outputs': sdn_output.get('mlp_outputs') # SDN forward 的 MLP 输出 (list)
        }
        # 如果只针对特定域，可能需要调整返回结构
        if domain_idx is not None:
             outputs['mlp_output'] = difln_output['mlp_output']
             # ... 可能需要调整 SDN/DDN 的返回

        return outputs

    def compute_losses(self, x, y, domains):
        # 执行一次前向传播获取所有中间结果
        outputs = self.forward(x, compute_all=True) # compute_all 标志可能不需要

        features = outputs['features']
        di_features = outputs['di_features']
        mlp_outputs = outputs['mlp_outputs'] # List of tensors [num_domains, batch_size, num_classes]
        di_domain_output = outputs['di_domain_output'] # For adversarial loss
        ddn_domain_output = outputs['ddn_domain_output'] # For DDN domain classification loss
        ddn_mi_loss = outputs['ddn_mi_loss']
        sdn_mi_losses = outputs['sdn_mi_losses'] # List of tensors

        # 1. DIFLN - 分类损失 (mlp_loss)
        mlp_loss = 0
        num_valid_domains_mlp = 0
        for d in range(self.num_domains):
            domain_mask = (domains == d)
            if domain_mask.sum() > 0:
                domain_mlp_output = mlp_outputs[d][domain_mask]
                domain_labels = y[domain_mask]
                mlp_loss += F.cross_entropy(domain_mlp_output, domain_labels)
                num_valid_domains_mlp += 1
        mlp_loss = mlp_loss / num_valid_domains_mlp if num_valid_domains_mlp > 0 else torch.tensor(0.0, device=x.device)

        # 2. DIFLN - GCN 损失 (gcn_loss)
        gcn_loss, _ = self.difln.compute_gcn_loss(di_features, y, domains)

        # 3. DIFLN - 一致性损失 (cr_loss) - 在这里计算
        cr_loss = 0
        num_valid_domains_cr = 0
        for d in range(self.num_domains):
             domain_mask = (domains == d)
             if domain_mask.sum() > 0:
                 domain_samples_raw = x[domain_mask] # 使用原始 x
                 # 增强
                 weak_aug_samples = weak_augmentation(domain_samples_raw)
                 strong_aug_samples = strong_augmentation(domain_samples_raw)
                 # 提取增强后特征
                 with torch.no_grad(): # 通常CR loss不反向传播到特征提取器
                     weak_features = self.feature_extractor(weak_aug_samples)
                     strong_features = self.feature_extractor(strong_aug_samples)
                 # 提取 di features
                 weak_di_features = self.difln.semantic_disentangler(weak_features)
                 strong_di_features = self.difln.semantic_disentangler(strong_features)
                 # 计算专家输出
                 expert_output = self.difln.mlp_classifiers[d](weak_di_features)
                 # 计算非专家集成输出
                 non_expert_outputs = []
                 for j in range(self.num_domains):
                     if j != d:
                         non_expert_output = self.difln.mlp_classifiers[j](strong_di_features)
                         non_expert_outputs.append(non_expert_output)
                 if non_expert_outputs: # 确保至少有一个非专家
                     non_expert_outputs = torch.stack(non_expert_outputs)
                     cr_loss += self.difln.consistency_loss(expert_output, non_expert_outputs)
                     num_valid_domains_cr += 1
        cr_loss = cr_loss / num_valid_domains_cr if num_valid_domains_cr > 0 else torch.tensor(0.0, device=x.device)


        # 4. DDN - 对抗损失 (adv_loss) - 使用 di_domain_output
        adv_loss = F.cross_entropy(di_domain_output, domains)

        # 5. DDN - 域分类损失 (dce_loss) - 使用 ddn_domain_output
        dce_loss = F.cross_entropy(ddn_domain_output, domains)

        # 6. DDN - 互信息损失 (mid_loss)
        mid_loss = ddn_mi_loss # 直接使用 forward 的结果

        # 7. SDN - 熵损失 (ent_loss) - 调用修正后的方法
        ent_loss = self.sdn.compute_entropy_loss(features, self.difln.mlp_classifiers, domains)

        # 8. SDN - 互信息损失 (mis_loss)
        mis_loss = sum(sdn_mi_losses) / len(sdn_mi_losses) if sdn_mi_losses else torch.tensor(0.0, device=x.device)


        losses = {
            'mlp_loss': mlp_loss, # DIFLN classification
            'gcn_loss': gcn_loss, # DIFLN GCN
            'cr_loss': cr_loss,   # DIFLN Consistency
            'adv_loss': adv_loss, # DDN Adversarial (GRL(f_di) -> F_d)
            'dce_loss': dce_loss, # DDN Domain Classification (f_dd -> F_d)
            'mid_loss': mid_loss, # DDN MI(f_di, f_dd)
            'ent_loss': ent_loss, # SDN Entropy Maximization
            'mis_loss': mis_loss  # SDN MI(f_di, f_si)
        }

        return losses

    def predict(self, x):
        # 提取特征
        features = self.feature_extractor(x) # 使用 D2IFLN 的 feature_extractor
        di_features = self.difln.semantic_disentangler(features)

        # 使用所有MLP分类器的平均输出
        outputs = []
        for classifier in self.difln.mlp_classifiers:
            outputs.append(classifier(di_features))

        # 计算平均预测
        avg_output = torch.stack(outputs).mean(dim=0)
        _, pred = torch.max(avg_output, 1)

        return pred

