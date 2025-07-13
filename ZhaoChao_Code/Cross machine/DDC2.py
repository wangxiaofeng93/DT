import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader_1d2
import resnet18_1d2 as models
import torch.nn as nn
import time
import numpy as np
import random
from utils import mmd_rbf_noaccelerate
import matplotlib
matplotlib.use('TkAgg')  # 非交互式后端，如果打算plt.show出窗口则改为 TkAgg
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # 添加3D绘图支持

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
momentum = 0.9
no_cuda = False
seed = 8
log_interval = 10
l2_decay = 5e-4

def train(model):
    src_iter = iter(src_loader)
    
    start=time.time()
    for i in range(1, iteration + 1):
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i - 1) % 100 == 0:
            print(f'learning rate{LEARNING_RATE: .4f}')

        optimizer = torch.optim.Adam([
            {'params': model.sharedNet.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, weight_decay=l2_decay)

        try:
            src_data, src_label = next(src_iter)
        except Exception as err:
            src_iter = iter(src_loader)
            src_data, src_label = next(src_iter)

        if cuda:
            src_data, src_label = src_data.cuda(), src_label.cuda()

        domian_index_1 = (src_label[:, 1] == 0)
        domian_index_2 = (src_label[:, 1] == 1)
        domian_index_3 = (src_label[:, 1] == 2)

        src_label=src_label[:,0]

        optimizer.zero_grad()

        src_pred,src_feature= model(src_data)

        feature1=src_feature[domian_index_1]
        feature2=src_feature[domian_index_2]
        feature3=src_feature[domian_index_3]

        s1, d = feature1.shape
        s2, d = feature2.shape
        s3, d = feature3.shape

        minlen=min([s1,s2,s3])

        feature1=feature1[:minlen,:]
        feature2=feature2[:minlen,:]
        feature3=feature3[:minlen,:]

        cls_loss = F.nll_loss(F.log_softmax(src_pred, dim=1), src_label)

        MMD_loss=mmd_rbf_noaccelerate(feature1, feature2) \
                 +mmd_rbf_noaccelerate(feature1, feature3) \
                 +mmd_rbf_noaccelerate(feature2, feature3)

        loss = cls_loss + MMD_loss
        loss.backward()
        optimizer.step()
        
        if i % log_interval == 0:
            print(f'Train iter: {i} [({100. * i / iteration:.0f}%)]\tLoss: {loss.item():.6f}\tsoft_Loss: {cls_loss.item():.6f}\tMMD_Loss: {MMD_loss:.6f}')

        if i % (log_interval * 10) == 0:
            # 使用test_model替代test_source
            train_correct, train_loss, _ = test_model(model, src_loader)
            # 使用test_model替代test_target
            test_correct, test_loss, _ = test_model(model, tgt_test_loader)

def test_model(model, test_loader, class_num=3, return_predictions=False, return_features=False):
    model.eval()
    test_loss = 0
    correct = 0
    class_correct = [0] * class_num
    class_total = [0] * class_num
    all_preds = []  # 收集所有预测结果
    all_labels = []  # 收集所有真实标签
    all_features = [] if return_features else None  # 新增特征收集
    all_domain_labels = [] if return_features else None  # 新增域标签收集

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            domain_labels = tgt_test_label[:, 1]  # 提取域标签
            tgt_test_label = tgt_test_label[:,0]  # 类别标签
            tgt_pred, feature = model(tgt_test_data)  # 获取特征
            
            # 收集特征和域标签
            if return_features:
                all_features.append(feature.cpu().numpy())
                all_domain_labels.append(domain_labels.cpu().numpy())
            
            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim=1), tgt_test_label, reduction='sum').item()
            pred = tgt_pred.data.max(1)[1]
            # 收集结果
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(tgt_test_label.cpu().numpy())

            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()

            # 遍历每个样本计算类别准确率
            for j in range(len(tgt_test_label)):
                label = tgt_test_label[j].item()  # 获取第j个样本的真实标签
                class_correct[label] += (pred[j] == label).item()  # 若预测正确则增加该类别的正确计数
                class_total[label] += 1  # 增加该类别的总样本计数

    # 计算每个类别的准确率
    class_accuracy = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                      for i in range(class_num)]
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)')
    print(f'Class Accuracy: Normal: {100*class_accuracy[0]:.2f}%, Inner Race: {100*class_accuracy[1]:.2f}%, Outer Race: {100*class_accuracy[2]:.2f}%')
    
    # 构建返回结果
    result = (correct, test_loss, class_accuracy)
    if return_predictions:
        result += (all_preds, all_labels)
    if return_features:
        all_features = np.concatenate(all_features) if all_features else None
        all_domain_labels = np.concatenate(all_domain_labels) if all_domain_labels else None
        result += (all_features, all_domain_labels)
    return result

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total:{total_num} Trainable:{trainable_num}')


# 新增函数：可视化多个域的特征，使用不同形状表示不同域
def visualize_features_multidomain(save_path, class_names=None, domain_names=None, show=True, save=False,
                                   n_components=3):
    """
    对多个域的特征进行可视化，使用不同形状表示不同域，不同颜色表示不同类别

    参数:
        save_path: 特征文件路径（.npz），包含features, class_labels和domain_labels
        class_names: 类别名称列表（可选）
        domain_names: 域名称列表（可选）
        show: 是否显示图形
        save: 是否保存图形
        n_components: 降维维度 (2 或 3, 默认为3)
    """
    data = np.load(save_path)
    features = data['features']
    labels = data['class_labels']
    domain_labels = data['domain_labels']  # 直接从文件加载域标签

    # 检查数据一致性
    assert len(features) == len(labels) == len(domain_labels), "特征、真实标签和域标签数量不一致"
    assert n_components in [2, 3], "n_components必须是2或3"

    # 使用t-SNE降维
    tsne = TSNE(n_components=n_components, random_state=42)
    features_reduced = tsne.fit_transform(features)

    # 创建图形
    if n_components == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(10, 8))

    # 获取唯一的域标签和类别标签
    unique_domains = np.unique(domain_labels)
    unique_classes = np.unique(labels)

    # 定义形状列表，每个域一个形状
    markers = ['o', 's', '^', 'D', '*', 'p', 'h', '8']  # 圆形，方形，三角形，菱形等
    # 定义颜色映射，用于类别
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_classes)))

    # 为每个域创建图例句柄
    domain_handles = []
    class_handles = []

    # 绘制每个域的点
    for domain_idx, domain in enumerate(unique_domains):
        # 获取当前域的所有样本索引
        domain_mask = (domain_labels == domain)
        domain_features = features_reduced[domain_mask]
        domain_class_labels = labels[domain_mask]

        # 绘制当前域的点
        for class_idx, class_label in enumerate(unique_classes):
            # 获取当前类别在当前域中的样本
            class_mask = (domain_class_labels == class_label)
            if not np.any(class_mask):
                continue
            points = domain_features[class_mask]

            # 绘制点
            if n_components == 3:
                scatter = ax.scatter(
                    points[:, 0], points[:, 1], points[:, 2],
                    marker=markers[domain_idx % len(markers)],  # 索引超出范围后从头取
                    c=[colors[class_idx]] * len(points),
                    s=40,  # 点的大小
                    alpha=0.6,
                    label=f'Domain {domain}-{class_names[class_label]}' if class_names else f'Domain {domain}-Class {class_label}'
                )
            else:  # 2D情况
                scatter = ax.scatter(
                    points[:, 0], points[:, 1],
                    marker=markers[domain_idx % len(markers)],
                    c=[colors[class_idx]] * len(points),
                    s=40,
                    alpha=0.6,
                    label=f'Domain {domain}-{class_names[class_label]}' if class_names else f'Domain {domain}-Class {class_label}'
                )

            # 保存图例句柄（每个域的第一个类别）
            if class_idx == 0:
                domain_handles.append(plt.Line2D([0], [0],
                                                 marker=markers[domain_idx % len(markers)],
                                                 color='w', markerfacecolor='gray',
                                                 markersize=10,
                                                 label=f'Domain {domain}' if domain_names is None else domain_names[
                                                     domain]))
            # 保存类别的图例句柄（每个类别的第一个域）
            if domain_idx == 0:
                class_handles.append(plt.Line2D([0], [0],
                                                marker='o',
                                                color='w', markerfacecolor=colors[class_idx],
                                                markersize=10,
                                                label=class_names[
                                                    class_label] if class_names else f'Class {class_label}'))

    # 添加图例
    domain_legend = ax.legend(handles=domain_handles, title="Domains", loc='upper left')
    ax.add_artist(domain_legend)  # 添加第二个图例需要先添加第一个
    ax.legend(handles=class_handles, title="Classes", loc='upper right')

    ax.set_title(f'{n_components}D Feature Distribution by Domain and Class')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    if n_components == 3:
        ax.set_zlabel('Dimension 3')
        # 调整视角以获得更好的3D视图
        ax.view_init(elev=20, azim=35)

    # 保存和显示
    if save:
        dim_suffix = "2d" if n_components == 2 else "3d"
        plot_path = save_path.replace('.npz', f'_{dim_suffix}_multidomain.png')
        plt.savefig(plot_path, dpi=300)
        print(f"{n_components}D multidomain feature visualization saved to {plot_path}")
    if show:
        plt.show(block=True)
    plt.close()


def plotly_visualize_features_multidomain_3d(save_path, class_names=None, domain_names=None, save_html=True):
    """
    使用Plotly创建交互式3D多域特征可视化

    参数:
        save_path: 特征文件路径（.npz）
        class_names: 类别名称列表
        domain_names: 域名称列表
        save_html: 是否保存为HTML文件
    """
    import plotly.graph_objects as go
    import plotly.express as px
    from sklearn.manifold import TSNE
    import numpy as np
    # 加载数据
    data = np.load(save_path)
    features = data['features']
    labels = data['class_labels']
    domain_labels = data['domain_labels']

    # 检查数据一致性
    assert len(features) == len(labels) == len(domain_labels), "特征、标签和域标签数量不一致"

    # t-SNE降维到3D
    tsne = TSNE(n_components=3, random_state=42)
    features_3d = tsne.fit_transform(features)

    # 获取唯一的域和类别
    unique_domains = np.unique(domain_labels)
    unique_classes = np.unique(labels)

    # 为域分配形状
    domain_symbols = ['circle', 'square', 'diamond', 'cross', 'x',
                      'triangle-up', 'triangle-down', 'pentagon', 'hexagram']

    # 为类别分配颜色
    colors = px.colors.qualitative.Plotly

    # 创建图形对象
    fig = go.Figure()

    # 为每个域添加轨迹
    for domain_idx, domain in enumerate(unique_domains):
        domain_mask = (domain_labels == domain)
        domain_points = features_3d[domain_mask]
        domain_class_labels = labels[domain_mask]

        # 为当前域中的每个类别添加子轨迹
        for class_idx, class_label in enumerate(unique_classes):
            class_mask = (domain_class_labels == class_label)
            if not np.any(class_mask):
                continue

            points = domain_points[class_mask]

            # 轨迹名称
            domain_name = domain_names[domain] if domain_names else f'Domain {domain}'
            class_name = class_names[class_label] if class_names else f'Class {class_label}'
            trace_name = f'{domain_name} - {class_name}'

            # 添加散点图轨迹
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                name=trace_name,
                marker=dict(
                    symbol=domain_symbols[domain_idx % len(domain_symbols)],
                    color=colors[class_idx % len(colors)],
                    size=5,
                    opacity=0.7,
                    line=dict(width=0)
                ),
                hovertemplate=f'<b>{trace_name}</b><br>' +
                              'Dim1: %{x:.2f}<br>' +
                              'Dim2: %{y:.2f}<br>' +
                              'Dim3: %{z:.2f}<extra></extra>'
            ))

    # 设置布局
    fig.update_layout(
        title='交互式3D特征分布 - 按域和类别',
        scene=dict(
            xaxis_title='维度 1',
            yaxis_title='维度 2',
            zaxis_title='维度 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # 初始视角
            )
        ),
        legend=dict(
            title=dict(text='<b>域 - 类别</b>'),
            itemsizing='constant',
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=800
    )

    # 保存和显示
    if save_html:
        html_path = save_path.replace('.npz', '_interactive_3d.html')
        fig.write_html(html_path, include_plotlyjs='cdn')
        print(f"交互式3D可视化已保存至: {html_path}")

    return fig

def plot_confusion_matrix_from_data(all_labels, all_preds, class_names, save_path=None):
    """
    直接从数据绘制混淆矩阵（无需模型和数据加载器）

    参数:
        all_labels: 真实标签列表
        all_preds: 预测标签列表
        class_names: 类别名称列表
        save_path: 保存路径（可选）
    """
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.close()

if __name__ == '__main__':
    iteration = 500
    batch_size = 256
    lr = 0.0001
    class_num = 3

    datasetlist = [
        ['CWRU_SELECT', 'HUST', 'JNU',  'MFPT'],
        ['CWRU_SELECT', 'JNU',  'MFPT', 'HUST'],
        ['CWRU_SELECT', 'MFPT', 'HUST', 'JNU' ],
        ['HUST', 'JNU',  'MFPT','CWRU_SELECT'],
    ]
    
    all_task_results = []
    class_names = ['Normal', 'Inner Race', 'Outer Race']
    
    for taskindex in range(len(datasetlist)):
        dataset1 = datasetlist[taskindex][0]
        dataset2 = datasetlist[taskindex][1]
        dataset3 = datasetlist[taskindex][2]
        dataset4 = datasetlist[taskindex][3]

        best_repeat = -1
        best_accuracy = 0.0
        best_model = None
        
        for repeat in range(1):
            cuda = not no_cuda and torch.cuda.is_available()
            current_seed = seed + repeat
            torch.manual_seed(current_seed)

            if cuda:
                torch.cuda.manual_seed(current_seed)

            kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

            src_loader = data_loader_1d2.load_training(dataset1, dataset2, dataset3, batch_size, kwargs)
            tgt_test_loader = data_loader_1d2.load_testing(dataset4, batch_size, kwargs)
            
            src_dataset_len = len(src_loader.dataset)  # 数据集的总样本数
            src_loader_len = len(src_loader)  # 数据加载器的批次数目 Batch count
            model = models.CNN_1D(num_classes=class_num)
            print(model)
            if cuda:
                model.cuda()
            train(model)

            # 训练后测试：使用test_model替代test_source/test_target
            src_result = test_model(
                model, src_loader, return_predictions=True, return_features=True
            )
            tgt_result = test_model(
                model, tgt_test_loader, return_predictions=True, return_features=True
            )
            
            # 解包结果
            src_correct, src_loss, src_class_acc, src_preds, src_true_labels, src_features, src_domain_labels = src_result
            tgt_correct, tgt_loss, tgt_class_acc, tgt_preds, tgt_true_labels, tgt_features, tgt_domain_labels = tgt_result
            
            tgt_accuracy = tgt_correct.item() / len(tgt_test_loader.dataset)
            
            if tgt_accuracy > best_accuracy:
                best_accuracy = tgt_accuracy
                best_repeat = repeat
                best_model = model.state_dict().copy()
                # 保存最佳结果
                best_src_data = (src_features, src_true_labels, src_domain_labels)
                best_tgt_data = (tgt_features, tgt_true_labels, tgt_domain_labels)
                best_tgt_results = (tgt_correct, tgt_loss, tgt_class_acc, tgt_preds, tgt_true_labels)
            
            print(f'\n{"="*50}')
            print(f'TaskIndex {taskindex} - Repeat {repeat} Final Results:')
            # 复用目标域测试结果，避免重复测试
            print(f'Target set: Accuracy: {tgt_correct.item()}/{len(tgt_test_loader.dataset)} '
                  f'({100. * tgt_accuracy:.2f}%)')

            all_task_results.append({
                'taskindex': taskindex,
                'repeat': repeat,
                'source_acc': src_correct.item() / len(src_loader.dataset),
                'source_class_acc': src_class_acc,
                'target_acc': tgt_accuracy,
                'target_class_acc': tgt_class_acc
            })

        if best_model is not None:
            # 保存特征数据
            save_dir = f"results/DDC/task_{taskindex}"
            os.makedirs(save_dir, exist_ok=True)

            cm_path = os.path.join(save_dir, "confusion_matrix.png")
            # 解包最佳repeat  预测标签 和 真实标签
            _, _, _, best_preds, best_true_labels= best_tgt_results
            # 直接使用保存的结果绘制混淆矩阵
            plot_confusion_matrix_from_data(
                best_true_labels,
                best_preds,
                class_names,
                cm_path
            )


            # 解包最佳repeat 源域数据
            best_src_features, best_src_true_labels, best_src_domain_labels = best_src_data
            # 解包最佳repeat 目标域数据
            best_tgt_features, best_tgt_true_labels, best_tgt_domain_labels = best_tgt_data

            # 保存源域特征（包含域标签）
            np.savez(os.path.join(save_dir, f"source_features.npz"),
                features=best_src_features,
                true_labels=best_src_true_labels,  # 源域使用真实标签
                domain_labels=best_src_domain_labels
            )
            
            # 保存目标域特征：使用预测标签作为类别标签
            np.savez(os.path.join(save_dir, f"target_features.npz"),
                features=best_tgt_features,
                true_labels=best_preds,  # 目标域使用预测标签
                domain_labels=best_tgt_domain_labels
            )

            # 合并所有域的特征用于可视化
            combined_features = np.vstack((best_src_features, best_tgt_features))
            combined_labels = np.concatenate((best_src_true_labels, best_preds))   # 源域使用真实标签，目标域使用预测标签
            combined_domains = np.concatenate((best_src_domain_labels, best_tgt_domain_labels))
            
            # 保存合并的特征
            combined_path = os.path.join(save_dir, "combined_features.npz")
            np.savez(combined_path,
                features=combined_features,
                class_labels=combined_labels,
                domain_labels=combined_domains
            )
            
            # 可视化多域特征
            domain_names = [dataset1, dataset2, dataset3, dataset4]
            visualize_features_multidomain(
                save_path=combined_path,
                class_names=class_names,
                domain_names=domain_names,
                show=False,
                save=True
            )


            print(f"Task {taskindex}: Best repeat {best_repeat} with accuracy {best_accuracy:.4f}")
        
    print('\n\n' + '='*50)
    print('Summary of Target Domain Results:')
    for t_index in range(len(datasetlist)):
        task_results = [r for r in all_task_results if r['taskindex'] == t_index]

        if not task_results:
            print(f'\nTaskIndex {t_index}: No results found')
            continue

        avg_acc = sum(r['target_acc'] for r in task_results) / len(task_results)
        avg_class_acc = [
            sum(r['target_class_acc'][i] for r in task_results) / len(task_results)
            for i in range(class_num)
        ]

        print(f'\nTaskIndex {t_index}:')
        print(f'  Avg Accuracy: {avg_acc:.4f}')
        print(f'  Class Accuracies:')
        print(f'    Normal: {avg_class_acc[0]:.4f}')
        print(f'    Inner Race: {avg_class_acc[1]:.4f}')
        print(f'    Outer Race: {avg_class_acc[2]:.4f}')

    # 保存所有任务结果到文件
    results_path = "results/DDC/all_task_results.npz"
    np.savez(results_path, all_task_results=all_task_results)
    print(f"\nAll task results saved to {results_path}")
