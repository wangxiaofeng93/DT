import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from D2IFLN import D2IFLN
import torchvision.datasets as datasets

# 设置参数
parser = argparse.ArgumentParser()
# 数据集设置
parser.add_argument('--data_root', type=str, default='./data', help='数据根目录')
parser.add_argument('--dataset', type=str, default='PACS', help='数据集: PACS, Office-Home, VLCS')
# 训练设置
parser.add_argument('--batch_size', type=int, default=30, help='批大小')
parser.add_argument('--epochs', type=int, default=40, help='训练轮数')
parser.add_argument('--lr', type=float, default=0.002, help='学习率')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD动量')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')
# 模型设置
parser.add_argument('--backbone', type=str, default='resnet18', help='backbone: resnet18, resnet50')
parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
parser.add_argument('--seed', type=int, default=42, help='随机种子')
# 损失权重
parser.add_argument('--lambda_gcn', type=float, default=1.0, help='GCN损失权重')
parser.add_argument('--lambda_cr', type=float, default=1.0, help='一致性损失权重')
parser.add_argument('--lambda_adv', type=float, default=0.1, help='对抗损失权重')
parser.add_argument('--lambda_ent', type=float, default=1.0, help='熵损失权重')
parser.add_argument('--lambda_mis', type=float, default=1.0, help='SDN MI损失权重')
parser.add_argument('--lambda_dce', type=float, default=1.0, help='域分类损失权重')
parser.add_argument('--lambda_mid', type=float, default=1.0, help='DDN MI损失权重')
args = parser.parse_args()

# 数据集配置
dataset_config = {
    'PACS': {
        'domains': ['photo', 'art', 'cartoon', 'sketch'],
        'num_classes': 7
    },
    'Office-Home': {
        'domains': ['art', 'clipart', 'product', 'real'],
        'num_classes': 65
    },
    'VLCS': {
        'domains': ['pascal', 'labelme', 'caltech', 'sun'],
        'num_classes': 5
    }
}

# 数据转换
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])


# 数据加载函数
def load_data(dataset_name, target_domain_idx):
    config = dataset_config[dataset_name]
    domains = config['domains']
    num_classes = config['num_classes']

    target_domain = domains[target_domain_idx]
    source_domains = [d for d in domains if d != target_domain]

    # 创建数据加载器
    source_loaders = []
    for i, domain in enumerate(source_domains):
        domain_path = os.path.join(args.data_root, dataset_name, domain)
        domain_dataset = datasets.ImageFolder(root=domain_path, transform=train_transform)
        domain_loader = DataLoader(
            domain_dataset, batch_size=args.batch_size // len(source_domains),
            shuffle=True, num_workers=4, drop_last=True
        )
        source_loaders.append((i, domain_loader))

    # 目标域数据
    target_path = os.path.join(args.data_root, dataset_name, target_domain)
    target_dataset = datasets.ImageFolder(root=target_path, transform=test_transform)
    target_loader = DataLoader(
        target_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4
    )

    return source_loaders, target_loader, num_classes, len(source_domains)


# 创建模型函数
def create_model(num_classes, num_domains):
    # 创建模型
    if args.backbone == 'resnet18':
        feature_extractor = models.resnet18(pretrained=True)
        feature_dim = 512
    elif args.backbone == 'resnet50':
        feature_extractor = models.resnet50(pretrained=True)
        feature_dim = 2048
    else:
        raise ValueError(f"不支持的backbone: {args.backbone}")

    # 移除分类层
    feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])
    feature_extractor.add_module('flatten', nn.Flatten())

    # 创建D²IFLN模型
    model = D2IFLN(
        feature_extractor=feature_extractor,
        feature_dim=feature_dim,
        num_classes=num_classes,
        num_domains=num_domains,
        hidden_dim=args.hidden_dim
    )

    # 转移到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model, device


# 修改训练函数以接受优化器并实现迭代训练
def train(model, optimizers, source_loaders, num_domains, device, epoch):
    model.train()
    total_loss_log = 0 # 用于日志记录的总损失

    difln_optimizer = optimizers['difln']
    sdn_stage1_optimizer = optimizers['sdn1']
    sdn_stage2_optimizer = optimizers['sdn2']
    ddn_stage1_optimizer = optimizers['ddn1']
    ddn_stage2_optimizer = optimizers['ddn2']

    # 获取数据迭代器
    iterators = [iter(loader) for _, loader in source_loaders]
    # 计算迭代次数，确保所有域的数据都被遍历大致相同的次数
    num_iters = min([len(loader) for _, loader in source_loaders])

    for iter_idx in range(num_iters):
        # 收集来自各个源域的样本
        all_x, all_y, all_domains = [], [], []
        for domain_idx, iterator in enumerate(iterators):
            try:
                x_batch, y_batch = next(iterator)
            except StopIteration:
                # 如果迭代器耗尽，重新初始化
                iterators[domain_idx] = iter(source_loaders[domain_idx][1])
                x_batch, y_batch = next(iterators[domain_idx])

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            all_x.append(x_batch)
            all_y.append(y_batch)
            # 为每个域的样本创建域标签
            all_domains.append(torch.full((x_batch.size(0),), source_loaders[domain_idx][0], dtype=torch.long, device=device)) # 使用传入的域索引

        # 合并来自所有源域的数据
        x = torch.cat(all_x, dim=0)
        y = torch.cat(all_y, dim=0)
        domains = torch.cat(all_domains, dim=0)

        # --- 算法描述的内部迭代过程 ---
        # 注意：每次 backward 后计算图会被释放，所以在每个优化步骤前需要重新计算损失
        # 这可能效率不高，更优化的方法是只计算当前阶段需要的损失并只反向传播该损失，
        # 但为清晰起见，这里每次都调用 compute_losses 获取所有损失值。

        # 1. 训练 DIFLN (更新 G, Dc, {F_mlp}, F_gcn, Fd)
        #    损失: mlp_loss, gcn_loss, cr_loss, adv_loss
        difln_optimizer.zero_grad()
        # 前向传播并计算所有损失
        losses = model.compute_losses(x, y, domains)
        difln_loss = losses['mlp_loss'] + \
                     args.lambda_gcn * losses['gcn_loss'] + \
                     args.lambda_cr * losses['cr_loss'] + \
                     args.lambda_adv * losses['adv_loss'] # 对抗损失影响 G, Dc, Fd
        difln_loss.backward()
        difln_optimizer.step()

        # 2. 训练 SDN - 第一阶段 (更新 {D_s^k})
        #    损失: ent_loss
        sdn_stage1_optimizer.zero_grad()
        # 需要重新计算损失，因为模型参数已更新
        losses = model.compute_losses(x, y, domains)
        sdn_stage1_loss = args.lambda_ent * losses['ent_loss']
        # 只反向传播此阶段的损失
        sdn_stage1_loss.backward()
        sdn_stage1_optimizer.step()

        # 3. 训练 SDN - 第二阶段 (更新 {Dc, D_s^k, M_s^k})
        #    损失: mis_loss
        sdn_stage2_optimizer.zero_grad()
        losses = model.compute_losses(x, y, domains)
        sdn_stage2_loss = args.lambda_mis * losses['mis_loss']
        sdn_stage2_loss.backward()
        sdn_stage2_optimizer.step()

        # 4. 训练 DDN - 第一阶段 (更新 {Dd, Fd})
        #    损失: dce_loss
        ddn_stage1_optimizer.zero_grad()
        losses = model.compute_losses(x, y, domains)
        ddn_stage1_loss = args.lambda_dce * losses['dce_loss']
        ddn_stage1_loss.backward()
        ddn_stage1_optimizer.step()

        # 5. 训练 DDN - 第二阶段 (更新 {Dc, Dd, Md})
        #    损失: mid_loss
        ddn_stage2_optimizer.zero_grad()
        losses = model.compute_losses(x, y, domains)
        ddn_stage2_loss = args.lambda_mid * losses['mid_loss']
        ddn_stage2_loss.backward()
        ddn_stage2_optimizer.step()

        # 累加用于日志记录的损失值（使用每个阶段实际反向传播的值）
        total_loss_log += (difln_loss.item() + sdn_stage1_loss.item() +
                           sdn_stage2_loss.item() + ddn_stage1_loss.item() +
                           ddn_stage2_loss.item())

        # --- 内部迭代过程结束 ---

    avg_loss = total_loss_log / num_iters
    print(f'Epoch {epoch}, Avg Iter Loss: {avg_loss:.4f}')
    return avg_loss


def collect_samples_from_domains(iterators, source_loaders, device):
    """从各个源域收集样本"""
    all_x, all_y, all_domains = [], [], []
    
    for domain_idx, iterator in enumerate(iterators):
        try:
            x, y = next(iterator)
        except StopIteration:
            # 如果迭代器耗尽，重新初始化
            iterators[domain_idx] = iter(source_loaders[domain_idx][1])
            x, y = next(iterators[domain_idx])
            
        x, y = x.to(device), y.to(device)
        all_x.append(x)
        all_y.append(y)
        all_domains.append(torch.full((x.size(0),), domain_idx, device=device))
        
    return all_x, all_y, all_domains


# 测试函数
def test(model, target_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in target_loader:
            x, y = x.to(device), y.to(device)
            pred = model.predict(x)

            correct += (pred == y).sum().item()
            total += y.size(0)

    accuracy = 100.0 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    return accuracy


# 修改完整的训练和测试函数
def train_and_evaluate(target_domain_idx):
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 加载数据
    source_loaders, target_loader, num_classes, num_source_domains = load_data(
        args.dataset, target_domain_idx
    )
    # 注意：模型初始化需要的是源域的数量
    model, device = create_model(num_classes, num_source_domains)

    # --- 创建所有优化器 ---
    # 1. DIFLN优化器
    difln_params = list(model.feature_extractor.parameters()) + \
                   list(model.difln.semantic_disentangler.parameters()) + \
                   list(model.difln.mlp_classifiers.parameters()) + \
                   list(model.difln.edge_network.parameters()) + \
                   list(model.difln.node_classifiers.parameters()) + \
                   list(model.ddn.domain_classifier.parameters()) # Fd 包含在内
    difln_optimizer = optim.SGD(difln_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 2. SDN第一阶段优化器
    sdn_stage1_params = list(model.sdn.domain_specific_disentanglers.parameters())
    sdn_stage1_optimizer = optim.SGD(sdn_stage1_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 3. SDN第二阶段优化器
    sdn_stage2_params = list(model.difln.semantic_disentangler.parameters()) + \
                        list(model.sdn.domain_specific_disentanglers.parameters()) + \
                        list(model.sdn.mi_estimators.parameters())
    sdn_stage2_optimizer = optim.SGD(sdn_stage2_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 4. DDN第一阶段优化器
    ddn_stage1_params = list(model.ddn.domain_disentangler.parameters()) + \
                        list(model.ddn.domain_classifier.parameters())
    ddn_stage1_optimizer = optim.SGD(ddn_stage1_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 5. DDN第二阶段优化器
    ddn_stage2_params = list(model.difln.semantic_disentangler.parameters()) + \
                        list(model.ddn.domain_disentangler.parameters()) + \
                        list(model.ddn.mi_estimator.parameters())
    ddn_stage2_optimizer = optim.SGD(ddn_stage2_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    optimizers = {
        'difln': difln_optimizer,
        'sdn1': sdn_stage1_optimizer,
        'sdn2': sdn_stage2_optimizer,
        'ddn1': ddn_stage1_optimizer,
        'ddn2': ddn_stage2_optimizer
    }
    # --- 优化器创建结束 ---

    # 学习率调度器 (应用到主优化器 difln_optimizer)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(difln_optimizer, T_max=args.epochs)

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        # 将所有优化器传递给 train 函数
        train_loss = train(model, optimizers, source_loaders, num_source_domains, device, epoch)
        # 对主优化器的学习率进行调整
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0] # 获取当前学习率
        # 可选：将其他优化器的学习率也同步调整（如果需要）
        for name, opt in optimizers.items():
             if name != 'difln':
                 for param_group in opt.param_groups:
                     param_group['lr'] = current_lr
        print(f"Epoch {epoch}, Current LR: {current_lr:.6f}")


        # 每个epoch结束后进行测试
        test_acc = test(model, target_loader, device)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            # 确保保存路径存在
            os.makedirs(os.path.dirname(f'./checkpoints/best_model_{args.dataset}_target_{target_domain_idx}.pth'), exist_ok=True)
            torch.save(model.state_dict(), f'./checkpoints/best_model_{args.dataset}_target_{target_domain_idx}.pth')


    print(f'目标域 {target_domain_idx} ({dataset_config[args.dataset]["domains"][target_domain_idx]}) 的最佳测试准确率: {best_acc:.2f}%')
    return best_acc


if __name__ == "__main__":
    print(f"训练数据集: {args.dataset}")
    print(f"backbone: {args.backbone}")
    print(f"参数: {args}") # 打印所有参数

    # 获取数据集的所有域
    domains = dataset_config[args.dataset]['domains']
    num_total_domains = len(domains) # 总域数

    # 实现完整的leave-one-domain-out策略
    all_accuracies = []

    # 遍历每个域作为目标域
    for target_idx in range(num_total_domains):
        print(f"\n{'=' * 50}")
        print(f"目标域: {domains[target_idx]} (索引 {target_idx})")
        source_domain_names = [domains[i] for i in range(num_total_domains) if i != target_idx]
        print(f"源域: {source_domain_names}")
        print(f"{'=' * 50}\n")

        # 训练并评估模型
        accuracy = train_and_evaluate(target_idx)
        all_accuracies.append(accuracy)

    # 计算并输出平均准确率
    avg_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0
    print(f"\n{'=' * 70}")
    print(f"所有域作为目标域的准确率: {all_accuracies}")
    print(f"平均准确率: {avg_accuracy:.2f}%")
    print(f"{'=' * 70}")

    # 将结果保存到文件
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    results_filename = os.path.join(results_dir, f"{args.dataset}_{args.backbone}_results.txt")

    with open(results_filename, "w") as f:
        f.write(f"数据集: {args.dataset}\n")
        f.write(f"Backbone: {args.backbone}\n")
        f.write(f"参数: {args}\n\n")

        for i, domain in enumerate(domains):
            f.write(f"目标域 {domain}: {all_accuracies[i]:.2f}%\n")

        f.write(f"\n平均准确率: {avg_accuracy:.2f}%\n")

    print(f"结果已保存到: {results_filename}")