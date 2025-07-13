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

from Utility import evaluate_model as test_model
from Utility import visualize_features_multidomain, plot_confusion_matrix_from_data

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
momentum = 0.9
no_cuda = False
seed = 8
log_interval = 10
l2_decay = 5e-4


def train(model):
    src_iter = iter(src_loader)

    start = time.time()
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

        src_label = src_label[:, 0]

        optimizer.zero_grad()

        src_pred, src_feature = model(src_data)

        feature1 = src_feature[domian_index_1]
        feature2 = src_feature[domian_index_2]
        feature3 = src_feature[domian_index_3]

        s1, d = feature1.shape
        s2, d = feature2.shape
        s3, d = feature3.shape

        minlen = min([s1, s2, s3])

        feature1 = feature1[:minlen, :]
        feature2 = feature2[:minlen, :]
        feature3 = feature3[:minlen, :]

        cls_loss = F.nll_loss(F.log_softmax(src_pred, dim=1), src_label)

        MMD_loss = mmd_rbf_noaccelerate(feature1, feature2) \
                   + mmd_rbf_noaccelerate(feature1, feature3) \
                   + mmd_rbf_noaccelerate(feature2, feature3)

        loss = cls_loss + MMD_loss
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print(
                f'Train iter: {i} [({100. * i / iteration:.0f}%)]\tLoss: {loss.item():.6f}\tsoft_Loss: {cls_loss.item():.6f}\tMMD_Loss: {MMD_loss:.6f}')

        if i % (log_interval * 10) == 0:
            # 使用test_model替代test_source
            train_correct, train_loss, _ = test_model(model, src_loader)
            # 使用test_model替代test_target
            test_correct, test_loss, _ = test_model(model, tgt_test_loader)


if __name__ == '__main__':
    iteration = 500
    batch_size = 256
    lr = 0.0001
    class_num = 3

    datasetlist = [
        ['CWRU_SELECT', 'HUST', 'JNU', 'MFPT'],
        ['CWRU_SELECT', 'JNU', 'MFPT', 'HUST'],
        ['CWRU_SELECT', 'MFPT', 'HUST', 'JNU'],
        ['HUST', 'JNU', 'MFPT', 'CWRU_SELECT'],
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

            tgt_accuracy = tgt_correct / len(tgt_test_loader.dataset)

            if tgt_accuracy > best_accuracy:
                best_accuracy = tgt_accuracy
                best_repeat = repeat
                best_model = model.state_dict().copy()
                # 保存最佳结果
                best_src_data = (src_features, src_true_labels, src_domain_labels)
                best_tgt_data = (tgt_features, tgt_true_labels, tgt_domain_labels)
                best_tgt_results = (tgt_correct, tgt_loss, tgt_class_acc, tgt_preds, tgt_true_labels)

            print(f'\n{"=" * 50}')
            print(f'TaskIndex {taskindex} - Repeat {repeat} Final Results:')
            # 复用目标域测试结果，避免重复测试
            print(f'Target set: Accuracy: {tgt_correct}/{len(tgt_test_loader.dataset)} '  f'({100. * tgt_accuracy:.2f}%)')
            print(
                f'Class Accuracy: Normal: {100 * tgt_class_acc[0]:.2f}%, Inner Race: {100 * tgt_class_acc[1]:.2f}%, Outer Race: {100 * tgt_class_acc[2]:.2f}%')

            all_task_results.append({
                'taskindex': taskindex,
                'repeat': repeat,
                'source_acc': src_correct / len(src_loader.dataset),
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
            _, _, _, best_preds, best_true_labels = best_tgt_results
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
            combined_labels = np.concatenate((best_src_true_labels, best_preds))  # 源域使用真实标签，目标域使用预测标签
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

    print('\n\n' + '=' * 50)
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
