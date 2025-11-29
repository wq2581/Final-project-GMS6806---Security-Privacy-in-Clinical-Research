import torch
from transformers import BertForSequenceClassification
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np

pretrained_model_name = "geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"


class GeneDataset(Dataset):
    """自定义Dataset，同时加载样本和标签"""
    def __init__(self, sample_path, label_path):
        self.samples = np.load(sample_path)
        self.labels = np.load(label_path)
        
        # 确保样本和标签数量一致
        assert len(self.samples) == len(self.labels), \
            f"Sample count {len(self.samples)} != Label count {len(self.labels)}"
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class Geneformer(nn.Module):
    def __init__(self, pretrained_path, hidden_layer=6):
        super(Geneformer, self).__init__()
        self.former = BertForSequenceClassification.from_pretrained(
            pretrained_path,
            num_labels=3,
            output_attentions=False,
            output_hidden_states=True
        )
        self.hidden_layer = hidden_layer

    def forward(self, seq):
        output = self.former(seq)
        # 提取指定层的hidden states
        hidden_states = output['hidden_states'][self.hidden_layer]
        # 对序列维度取平均，得到句子级别的表示
        x = torch.mean(hidden_states, dim=1)
        return x


def generate_embeddings_with_labels(
    samples_dir="./samples/",
    labels_dir="./labels/",
    output_dir="./paired_data/",
    pretrained_path="geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/",
    batch_size=2,
    hidden_layer=6,
    device=None
):
    """
    生成 embedding 和 label 配对的数据集
    
    Args:
        samples_dir: 样本目录
        labels_dir: 标签目录
        output_dir: 输出目录
        pretrained_path: 预训练模型路径
        batch_size: 批次大小
        hidden_layer: 使用哪一层的hidden states
        device: 计算设备
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 加载模型
    print("Loading Geneformer model...")
    model = Geneformer(pretrained_path, hidden_layer=hidden_layer)
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    # 获取所有样本文件
    sample_files = [f for f in os.listdir(samples_dir) if f.endswith('_samples.npy')]
    
    if not sample_files:
        raise ValueError(f"No sample files found in {samples_dir}")
    
    print(f"Found {len(sample_files)} sample files")
    
    # 处理每个文件
    for sample_file in sample_files:
        # 构建对应的标签文件名
        base_name = sample_file.replace('_samples.npy', '')
        label_file = base_name + '_labels.npy'
        
        sample_path = os.path.join(samples_dir, sample_file)
        label_path = os.path.join(labels_dir, label_file)
        
        # 检查标签文件是否存在
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found for {sample_file}, skipping...")
            continue
        
        print(f"\nProcessing: {base_name}")
        
        # 创建数据集和数据加载器
        dataset = GeneDataset(sample_path, label_path)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        print(f"  Total samples: {len(dataset)}")
        
        # 生成embeddings
        all_embeddings = []
        all_labels = []
        
        pbar = tqdm(dataloader, desc=f"  Generating embeddings")
        with torch.no_grad():
            for samples, labels in pbar:
                # 将样本移到设备上
                samples = samples.to(device).long()
                
                # 生成embeddings
                embeddings = model(samples)
                
                # 收集结果
                all_embeddings.append(embeddings.detach().cpu().numpy())
                all_labels.append(labels.numpy())
        
        # 合并所有批次
        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.concatenate(all_labels)
        
        # 保存配对的数据
        embed_output_path = os.path.join(output_dir, f'{base_name}_embeds.npy')
        label_output_path = os.path.join(output_dir, f'{base_name}_labels.npy')
        
        np.save(embed_output_path, all_embeddings)
        np.save(label_output_path, all_labels)
        
        print(f"  Saved embeddings: {embed_output_path} (shape: {all_embeddings.shape})")
        print(f"  Saved labels: {label_output_path} (shape: {all_labels.shape})")
        print(f"  Label distribution: {np.bincount(all_labels.astype(int))}")


def load_paired_dataset(base_name, data_dir="./paired_data/"):
    """
    加载配对的embedding和label数据
    
    Args:
        base_name: 数据集基础名称（不含后缀）
        data_dir: 数据目录
    
    Returns:
        embeddings, labels
    """
    embed_path = os.path.join(data_dir, f'{base_name}_embeds.npy')
    label_path = os.path.join(data_dir, f'{base_name}_labels.npy')
    
    embeddings = np.load(embed_path)
    labels = np.load(label_path)
    
    assert len(embeddings) == len(labels), "Embeddings and labels length mismatch"
    
    return embeddings, labels


class PairedEmbeddingDataset(Dataset):
    """用于训练下游任务的配对数据集"""
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


if __name__ == "__main__":
    # 生成配对的embeddings和labels
    generate_embeddings_with_labels(
        samples_dir="./samples/",
        labels_dir="./labels/",
        output_dir="./paired_data/",
        pretrained_path="geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/",
        batch_size=32,  # 可以根据GPU内存调整
        hidden_layer=6,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\n" + "="*60)
    print("Embedding generation completed!")
    print("="*60)
    
    # 示例：如何加载和使用配对数据
    print("\n示例：加载配对数据")
    data_files = [f.replace('_embeds.npy', '') 
                  for f in os.listdir('./paired_data/') 
                  if f.endswith('_embeds.npy')]
    
    if data_files:
        example_file = data_files[0]
        embeddings, labels = load_paired_dataset(example_file, data_dir='./paired_data/')
        print(f"\nLoaded {example_file}:")
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Label distribution: {np.bincount(labels.astype(int))}")
        
        # 创建PyTorch数据集
        paired_dataset = PairedEmbeddingDataset(embeddings, labels)
        paired_loader = DataLoader(paired_dataset, batch_size=16, shuffle=True)
        
        print(f"\nDataLoader created with {len(paired_dataset)} samples")
        
        # 测试加载一个batch
        for batch_embeds, batch_labels in paired_loader:
            print(f"  Batch embeddings shape: {batch_embeds.shape}")
            print(f"  Batch labels shape: {batch_labels.shape}")
            break
