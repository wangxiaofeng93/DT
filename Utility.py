# Utility.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        # 最大化熵等价于最小化负熵
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1)
        return b.mean()


class MutualInformationEstimator(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim):
        super(MutualInformationEstimator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim1 + input_dim2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, z):
        # Concat x and z
        batch_size = x.size(0)
        input_tensor = torch.cat([x, z], dim=1)

        # Joint distribution
        joint = self.network(input_tensor)

        # Marginal distribution (shuffle z)
        z_shuffle = z[torch.randperm(batch_size)]
        marginal_input = torch.cat([x, z_shuffle], dim=1)
        marginal = self.network(marginal_input)

        # Mutual information estimation (MINE)
        mi_estimation = joint.mean() - torch.log(torch.exp(marginal).mean())
        return mi_estimation


def mutual_information_loss(mi_estimator, x, z):
    # Minimize mutual information
    return mi_estimator(x, z)


class ConsistencyRegLoss(nn.Module):
    def __init__(self):
        super(ConsistencyRegLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, expert_output, non_expert_outputs):
        # 计算一个域的专家与其他域非专家集成之间的MSE
        non_expert_ensemble = torch.mean(non_expert_outputs, dim=0)
        return self.mse(expert_output, non_expert_ensemble)


# 添加梯度反转层
class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)


def evaluate_model(model, data_loader, class_num=3, return_predictions=False, return_features=False):
    """
    通用模型评估函数

    参数:
        model: 要评估的模型（统一输出格式）
        data_loader: 数据加载器
        class_num: 类别数量
        return_predictions: 是否返回预测结果
        return_features: 是否返回特征
        device: 计算设备

    返回:
        (correct, test_loss, class_accuracy, [all_preds], [all_labels], [features], [domain_labels])
    """
    # 自动检测并选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.eval()
    test_loss = 0
    correct = 0
    class_correct = [0] * class_num
    class_total = [0] * class_num
    all_preds = []
    all_labels = []
    all_features = [] if return_features else None
    all_domain_labels = [] if return_features else None

    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            labels = labels.to(device)

            # 统一模型输出接口
            outputs = model(data)
            preds = outputs[0]  # 分类结果
            features = outputs[1] if len(outputs) > 1 else None  # 特征
            domain_preds = outputs[2] if len(outputs) > 2 else None  # 域预测（可选）

            # 提取域标签（如果存在）
            if labels.dim() > 1 and labels.shape[1] > 1:
                domain_labels = labels[:, 1]
            else:
                domain_labels = None

            # 收集特征
            if return_features and features is not None:
                all_features.append(features.cpu().numpy())
                if domain_labels is not None:
                    all_domain_labels.append(domain_labels.cpu().numpy())

            # 计算损失和准确率
            test_loss += F.nll_loss(F.log_softmax(preds, dim=1), labels[:, 0], reduction='sum').item()
            pred = preds.argmax(dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels[:, 0].cpu().numpy())
            correct += pred.eq(labels[:, 0]).sum().item()

            # 计算每个类别的准确率
            for i in range(len(labels)):
                label = labels[i, 0].item()
                class_correct[label] += (pred[i] == label).item()
                class_total[label] += 1

    # 计算每个类别的准确率
    class_accuracy = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                      for i in range(class_num)]

    print(f'\n Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)')
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


# 添加可视化函数导入
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，如果打算plt.show出窗口则改为 TkAgg
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
import seaborn as sns


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
        fig, ax = plt.subplots(figsize=(12, 10))
    plt.tight_layout(rect=[0.02, 0.02, 0.85, 0.98])  # 保留5%的边距

    # 获取唯一的域标签和类别标签
    unique_domains = np.unique(domain_labels)
    unique_classes = np.unique(labels)

    # 定义形状列表，每个域一个形状
    markers = ['o', 's', '^', 'D', '*', 'p', 'h', '8']  # 圆形，方形，三角形，菱形等
    # 定义颜色映射，用于类别
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_classes)))

    # 创建组合图例句柄和标签
    legend_handles = []
    legend_labels = []

    # 绘制每个域的点
    for domain_idx, domain in enumerate(unique_domains):
        # 获取当前域的所有样本索引
        domain_mask = (domain_labels == domain)
        domain_features = features_reduced[domain_mask]
        domain_class_labels = labels[domain_mask]

        # 获取当前域的名称
        domain_name = f'Domain {domain}' if domain_names is None else domain_names[domain]
        marker = markers[domain_idx % len(markers)]  # 索引超出范围后从头取

        # 绘制当前域中每个类别的点
        for class_idx, class_label in enumerate(unique_classes):
            # 获取当前类别在当前域中的样本
            class_mask = (domain_class_labels == class_label)
            if not np.any(class_mask):
                continue
            points = domain_features[class_mask]

            # 获取当前类别的名称和颜色
            class_name = class_names[class_label] if class_names else f'Class {class_label}'
            color = colors[class_idx]
            label = f'{domain_name} - {class_name}'

            # 绘制点
            if n_components == 3:
                scatter = ax.scatter(
                    points[:, 0], points[:, 1], points[:, 2],
                    marker=marker,
                    c=[color] * len(points),
                    s=40,  # 点的大小
                    alpha=0.6,
                    label=label
                )
            else:  # 2D情况
                scatter = ax.scatter(
                    points[:, 0], points[:, 1],
                    marker=marker,
                    c=[color] * len(points),
                    s=40,
                    alpha=0.6,
                    label=label
                )

            # 保存图例句柄和标签（每个组合只添加一次）
            legend_handles.append(scatter)
            legend_labels.append(label)

    # 添加组合图例
    ax.legend(handles=legend_handles, labels=legend_labels, title="Domain - Class",
              loc='center left', bbox_to_anchor=(1, 0.5),
              ncol=1 if len(legend_handles) < 15 else 2)  # 根据项数调整列数

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
