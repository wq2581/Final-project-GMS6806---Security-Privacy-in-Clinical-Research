import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import copy

# 评估指标函数
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score


class PairedEmbeddingDataset(Dataset):
    """配对的embedding和label数据集"""
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class MLP_Classifier(nn.Module):
    def __init__(self, input_size=256, hidden_size=512, num_classes=2):
        super(MLP_Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def compute_metrics(pred, labels):
    """计算所有评估指标"""
    pred_probs = torch.softmax(pred, dim=1)[:, 1].detach().cpu().numpy()
    pred_labels = torch.argmax(pred, dim=1).detach().cpu().numpy()
    true_labels = labels.detach().cpu().numpy()
    
    metrics = {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'f1': f1_score(true_labels, pred_labels, average='binary', zero_division=0),
        'precision': precision_score(true_labels, pred_labels, average='binary', zero_division=0),
        'recall': recall_score(true_labels, pred_labels, average='binary', zero_division=0),
    }
    
    try:
        metrics['auroc'] = roc_auc_score(true_labels, pred_probs)
    except:
        metrics['auroc'] = 0.0
    
    return metrics, pred_probs, pred_labels


def load_all_datasets(data_dir='./paired_data/'):
    """加载所有配对的数据集"""
    embed_files = [f for f in os.listdir(data_dir) if f.endswith('_embeds.npy')]
    
    datasets = {}
    for embed_file in embed_files:
        base_name = embed_file.replace('_embeds.npy', '')
        label_file = base_name + '_labels.npy'
        
        embed_path = os.path.join(data_dir, embed_file)
        label_path = os.path.join(data_dir, label_file)
        
        embeddings = np.load(embed_path)
        labels = np.load(label_path)
        
        datasets[base_name] = {
            'embeddings': embeddings,
            'labels': labels
        }
        print(f"Loaded {base_name}: {embeddings.shape[0]} samples, "
              f"Label dist: {np.bincount(labels.astype(int))}")
    
    return datasets


def split_dataset(embeddings, labels, train_rate=0.8, seed=42):
    """划分训练集和测试集"""
    dataset = PairedEmbeddingDataset(embeddings, labels)
    train_size = int(len(dataset) * train_rate)
    test_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=generator
    )
    
    return train_dataset, test_dataset


# ==================== 方法1: Baseline ====================
def train_baseline(train_loader, test_loader, args, device):
    """标准训练（Baseline）"""
    print("\n" + "="*60)
    print("Training: BASELINE")
    print("="*60)
    
    model = MLP_Classifier(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_classes=args.num_classes
    ).to(device)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    history = {'train': [], 'test': []}
    
    for epoch in range(args.epochs):
        # 训练
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        for embeds, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            embeds, labels = embeds.to(device), labels.long().to(device)
            
            pred = model(embeds)
            loss = loss_function(pred, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.append(pred.detach())
            train_labels.append(labels.detach())
        
        train_preds = torch.cat(train_preds)
        train_labels = torch.cat(train_labels)
        train_metrics, _, _ = compute_metrics(train_preds, train_labels)
        train_metrics['loss'] = train_loss / len(train_loader)
        
        # 测试
        model.eval()
        test_loss = 0
        test_preds, test_labels = [], []
        
        with torch.no_grad():
            for embeds, labels in test_loader:
                embeds, labels = embeds.to(device), labels.long().to(device)
                pred = model(embeds)
                loss = loss_function(pred, labels)
                
                test_loss += loss.item()
                test_preds.append(pred.detach())
                test_labels.append(labels.detach())
        
        test_preds = torch.cat(test_preds)
        test_labels = torch.cat(test_labels)
        test_metrics, test_probs, test_pred_labels = compute_metrics(test_preds, test_labels)
        test_metrics['loss'] = test_loss / len(test_loader)
        
        history['train'].append(train_metrics)
        history['test'].append(test_metrics)
        
        print(f"Epoch {epoch+1} - Train Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, AUROC: {train_metrics['auroc']:.4f} | "
              f"Test Loss: {test_metrics['loss']:.4f}, "
              f"Acc: {test_metrics['accuracy']:.4f}, AUROC: {test_metrics['auroc']:.4f}")
    
    return model, history, test_probs, test_pred_labels, test_labels.cpu().numpy()


# ==================== 方法2: DP-SGD ====================
def train_dp_sgd(train_loader, test_loader, args, device):
    """使用差分隐私训练"""
    print("\n" + "="*60)
    print("Training: DP-SGD")
    print("="*60)
    
    model = MLP_Classifier(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_classes=args.num_classes
    ).to(device)
    
    # 验证模型是否兼容Opacus
    model = ModuleValidator.fix(model)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # 配置隐私引擎
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=args.epochs,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        max_grad_norm=args.max_grad_norm,
    )
    
    print(f"Privacy Budget Target: ε={args.target_epsilon}, δ={args.target_delta}")
    print(f"Noise Multiplier: {optimizer.noise_multiplier:.4f}")
    print(f"Max Grad Norm: {args.max_grad_norm}")
    
    history = {'train': [], 'test': [], 'epsilon': []}
    
    for epoch in range(args.epochs):
        # 训练
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        for embeds, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            embeds, labels = embeds.to(device), labels.long().to(device)
            
            pred = model(embeds)
            loss = loss_function(pred, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.append(pred.detach())
            train_labels.append(labels.detach())
        
        train_preds = torch.cat(train_preds)
        train_labels = torch.cat(train_labels)
        train_metrics, _, _ = compute_metrics(train_preds, train_labels)
        train_metrics['loss'] = train_loss / len(train_loader)
        
        # 获取当前隐私消耗
        epsilon = privacy_engine.get_epsilon(args.target_delta)
        train_metrics['epsilon'] = epsilon
        
        # 测试
        model.eval()
        test_loss = 0
        test_preds, test_labels = [], []
        
        with torch.no_grad():
            for embeds, labels in test_loader:
                embeds, labels = embeds.to(device), labels.long().to(device)
                pred = model(embeds)
                loss = loss_function(pred, labels)
                
                test_loss += loss.item()
                test_preds.append(pred.detach())
                test_labels.append(labels.detach())
        
        test_preds = torch.cat(test_preds)
        test_labels = torch.cat(test_labels)
        test_metrics, test_probs, test_pred_labels = compute_metrics(test_preds, test_labels)
        test_metrics['loss'] = test_loss / len(test_loader)
        test_metrics['epsilon'] = epsilon
        
        history['train'].append(train_metrics)
        history['test'].append(test_metrics)
        history['epsilon'].append(epsilon)
        
        print(f"Epoch {epoch+1} - ε={epsilon:.2f}, δ={args.target_delta:.2e} | "
              f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, AUROC: {train_metrics['auroc']:.4f} | "
              f"Test Loss: {test_metrics['loss']:.4f}, "
              f"Acc: {test_metrics['accuracy']:.4f}, AUROC: {test_metrics['auroc']:.4f}")
    
    # 最终隐私预算
    final_epsilon = privacy_engine.get_epsilon(args.target_delta)
    print(f"\n{'='*60}")
    print(f"Final Privacy Budget Consumed: ε={final_epsilon:.2f}, δ={args.target_delta:.2e}")
    print(f"{'='*60}")
    
    return model, history, test_probs, test_pred_labels, test_labels.cpu().numpy()


# ==================== 方法3: FedAvg ====================
def train_fedavg(datasets, test_data, args, device):
    """联邦学习 FedAvg"""
    print("\n" + "="*60)
    print("Training: FedAvg")
    print("="*60)
    
    # 全局模型
    global_model = MLP_Classifier(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_classes=args.num_classes
    ).to(device)
    
    loss_function = nn.CrossEntropyLoss()
    
    # 为每个数据集创建数据加载器
    client_loaders = []
    for dataset_name, data in datasets.items():
        train_dataset, _ = split_dataset(
            data['embeddings'], 
            data['labels'], 
            train_rate=1.0,  # 全部用于训练
            seed=args.seed
        )
        loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True
        )
        client_loaders.append((dataset_name, loader))
        print(f"  Client {dataset_name}: {len(train_dataset)} samples")
    
    # 测试数据加载器
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)
    
    history = {'train': [], 'test': []}
    
    for round_idx in range(args.fed_rounds):
        print(f"\n--- Round {round_idx+1}/{args.fed_rounds} ---")
        
        local_weights = []
        local_losses = []
        
        # 本地训练
        for client_name, train_loader in client_loaders:
            # 创建本地模型（复制全局模型）
            local_model = MLP_Classifier(
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                num_classes=args.num_classes
            ).to(device)
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            
            optimizer = optim.AdamW(local_model.parameters(), lr=args.lr)
            
            # 本地训练
            local_model.train()
            epoch_loss = 0
            for _ in range(args.local_epochs):
                for embeds, labels in train_loader:
                    embeds, labels = embeds.to(device), labels.long().to(device)
                    
                    pred = local_model(embeds)
                    loss = loss_function(pred, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
            
            local_weights.append(copy.deepcopy(local_model.state_dict()))
            local_losses.append(epoch_loss / len(train_loader))
            print(f"  {client_name} - Loss: {epoch_loss/len(train_loader):.4f}")
        
        # FedAvg 聚合
        global_weights = {}
        for key in global_model.state_dict().keys():
            global_weights[key] = torch.stack([
                local_weights[i][key].float() for i in range(len(local_weights))
            ]).mean(0)
        
        global_model.load_state_dict(global_weights)
        
        # 全局测试
        global_model.eval()
        test_loss = 0
        test_preds, test_labels = [], []
        
        with torch.no_grad():
            for embeds, labels in test_loader:
                embeds, labels = embeds.to(device), labels.long().to(device)
                pred = global_model(embeds)
                loss = loss_function(pred, labels)
                
                test_loss += loss.item()
                test_preds.append(pred.detach())
                test_labels.append(labels.detach())
        
        test_preds = torch.cat(test_preds)
        test_labels = torch.cat(test_labels)
        test_metrics, test_probs, test_pred_labels = compute_metrics(test_preds, test_labels)
        test_metrics['loss'] = test_loss / len(test_loader)
        
        train_metrics = {'loss': np.mean(local_losses)}
        history['train'].append(train_metrics)
        history['test'].append(test_metrics)
        
        print(f"Global Test - Loss: {test_metrics['loss']:.4f}, "
              f"Acc: {test_metrics['accuracy']:.4f}, AUROC: {test_metrics['auroc']:.4f}")
    
    return global_model, history, test_probs, test_pred_labels, test_labels.cpu().numpy()


# ==================== 可视化 ====================
def plot_comparison_results(results, output_dir='./comparison_results/'):
    """绘制对比结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    methods = list(results.keys())
    metrics = ['accuracy', 'auroc', 'f1', 'precision', 'recall']
    
    # 1. Bar chart - 性能对比
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.25
    
    colors = {'Baseline': '#3498db', 'DP-SGD': '#e74c3c', 'FedAvg': '#2ecc71'}
    
    for idx, method in enumerate(methods):
        values = [results[method]['final_metrics'][m] for m in metrics]
        ax.bar(x + idx*width, values, width, label=method, color=colors.get(method, 'gray'))
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: Drug Response Prediction', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison_bar.png'), dpi=300)
    plt.close()
    
    # 2. ROC curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for method in methods:
        fpr, tpr, _ = roc_curve(
            results[method]['true_labels'], 
            results[method]['pred_probs']
        )
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{method} (AUC = {roc_auc:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300)
    plt.close()
    
    # 3. Training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    plot_metrics = ['loss', 'accuracy', 'auroc', 'f1']
    for idx, metric in enumerate(plot_metrics):
        ax = axes[idx // 2, idx % 2]
        
        for method in methods:
            if metric in results[method]['history']['test'][0]:
                values = [epoch[metric] for epoch in results[method]['history']['test']]
                ax.plot(values, label=method, marker='o', linewidth=2)
        
        ax.set_xlabel('Epoch/Round', fontsize=10, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=10, fontweight='bold')
        ax.set_title(f'{metric.upper()} over Training', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300)
    plt.close()
    
    # 4. Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, method in enumerate(methods):
        cm = confusion_matrix(
            results[method]['true_labels'], 
            results[method]['pred_labels']
        )
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar_kws={'label': 'Count'})
        axes[idx].set_title(f'{method}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('True', fontsize=10)
        axes[idx].set_xticklabels(['Resistant', 'Sensitive'])
        axes[idx].set_yticklabels(['Resistant', 'Sensitive'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300)
    plt.close()
    
    # 5. Privacy budget tracking (if DP-SGD is included)
    if 'DP-SGD' in results and 'epsilon' in results['DP-SGD']['history']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epsilon_values = results['DP-SGD']['history']['epsilon']
        epochs = range(1, len(epsilon_values) + 1)
        
        ax.plot(epochs, epsilon_values, marker='o', linewidth=2, 
                color='#e74c3c', label='ε (Privacy Budget)')
        ax.axhline(y=results['DP-SGD']['history']['epsilon'][-1], 
                  color='red', linestyle='--', linewidth=1.5, 
                  label=f'Final ε = {epsilon_values[-1]:.2f}')
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Privacy Budget (ε)', fontsize=12, fontweight='bold')
        ax.set_title('Privacy Budget Consumption over Training', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'privacy_budget_tracking.png'), dpi=300)
        plt.close()
        
        print(f"Privacy budget tracking plot saved")
    
    print(f"\nAll plots saved to {output_dir}")


def save_results_table(results, output_dir='./comparison_results/'):
    """保存结果表格"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建结果DataFrame
    data = []
    for method, res in results.items():
        row = {'Method': method}
        row.update(res['final_metrics'])
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.round(4)
    
    # 保存CSV
    csv_path = os.path.join(output_dir, 'results_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    
    # 保存详细结果
    with open(os.path.join(output_dir, 'detailed_results.pkl'), 'wb') as f:
        pickle.dump(results, f)


def main():
    parser = argparse.ArgumentParser(description='Drug Response Prediction Comparison')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./paired_data/',
                       help='Directory containing paired embeddings and labels')
    parser.add_argument('--output_dir', type=str, default='./comparison_results/',
                       help='Output directory for results')
    
    # 模型参数
    parser.add_argument('--input_size', type=int, default=256,
                       help='Input embedding size')
    parser.add_argument('--hidden_size', type=int, default=512,
                       help='Hidden layer size')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs for Baseline and DP-SGD')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--test_batch_size', type=int, default=64,
                       help='Test batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--train_rate', type=float, default=0.8,
                       help='Training data ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # DP-SGD 参数
    parser.add_argument('--target_epsilon', type=float, default=10.0,
                       help='Target epsilon for DP-SGD')
    parser.add_argument('--target_delta', type=float, default=1e-5,
                       help='Target delta for DP-SGD')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Max gradient norm for DP-SGD')
    
    # FedAvg 参数
    parser.add_argument('--fed_rounds', type=int, default=20,
                       help='Number of federated learning rounds')
    parser.add_argument('--local_epochs', type=int, default=5,
                       help='Local epochs per round in FedAvg')
    
    # 实验选择
    parser.add_argument('--methods', nargs='+', 
                       default=['baseline', 'dp-sgd', 'fedavg'],
                       choices=['baseline', 'dp-sgd', 'fedavg'],
                       help='Methods to compare')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 加载所有数据集
    print("\nLoading datasets...")
    datasets = load_all_datasets(args.data_dir)
    
    if not datasets:
        raise ValueError(f"No datasets found in {args.data_dir}")
    
    # 合并所有数据用于 Baseline 和 DP-SGD
    all_embeddings = np.vstack([d['embeddings'] for d in datasets.values()])
    all_labels = np.concatenate([d['labels'] for d in datasets.values()])
    
    print(f"\nTotal dataset: {all_embeddings.shape[0]} samples")
    print(f"Label distribution: {np.bincount(all_labels.astype(int))}")
    
    # 划分训练集和测试集
    train_dataset, test_dataset = split_dataset(
        all_embeddings, all_labels, 
        train_rate=args.train_rate, 
        seed=args.seed
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.test_batch_size, 
        shuffle=False
    )
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    # 存储结果
    results = {}
    
    # 训练各个方法
    if 'baseline' in args.methods:
        model, history, probs, pred_labels, true_labels = train_baseline(
            train_loader, test_loader, args, device
        )
        results['Baseline'] = {
            'model': model,
            'history': history,
            'pred_probs': probs,
            'pred_labels': pred_labels,  # Already numpy array
            'true_labels': true_labels,  # Already numpy array
            'final_metrics': history['test'][-1]
        }
        
        # 保存模型
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), 
                  os.path.join(args.output_dir, 'baseline_model.pth'))
    
    if 'dp-sgd' in args.methods:
        model, history, probs, pred_labels, true_labels = train_dp_sgd(
            train_loader, test_loader, args, device
        )
        results['DP-SGD'] = {
            'model': model,
            'history': history,
            'pred_probs': probs,
            'pred_labels': pred_labels,  # Already numpy array
            'true_labels': true_labels,  # Already numpy array
            'final_metrics': history['test'][-1]
        }
        
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), 
                  os.path.join(args.output_dir, 'dpsgd_model.pth'))
    
    if 'fedavg' in args.methods:
        model, history, probs, pred_labels, true_labels = train_fedavg(
            datasets, test_dataset, args, device
        )
        results['FedAvg'] = {
            'model': model,
            'history': history,
            'pred_probs': probs,
            'pred_labels': pred_labels,  # Already numpy array
            'true_labels': true_labels,  # Already numpy array
            'final_metrics': history['test'][-1]
        }
        
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), 
                  os.path.join(args.output_dir, 'fedavg_model.pth'))
    
    # 可视化和保存结果
    plot_comparison_results(results, args.output_dir)
    save_results_table(results, args.output_dir)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED!")
    print("="*60)


if __name__ == '__main__':
    main()