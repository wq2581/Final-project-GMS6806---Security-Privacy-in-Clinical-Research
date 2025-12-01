import torch
from transformers import BertForSequenceClassification
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np

pretrained_model_name = "geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"


class GeneDataset(Dataset):

    def __init__(self, sample_path, label_path):
        self.samples = np.load(sample_path)
        self.labels = np.load(label_path)
        
     
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
     
        hidden_states = output['hidden_states'][self.hidden_layer]
    
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

    os.makedirs(output_dir, exist_ok=True)
    
 
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    

    print("Loading Geneformer model...")
    model = Geneformer(pretrained_path, hidden_layer=hidden_layer)
    model = model.to(device)
    model.eval()  

    sample_files = [f for f in os.listdir(samples_dir) if f.endswith('_samples.npy')]
    
    if not sample_files:
        raise ValueError(f"No sample files found in {samples_dir}")
    
    print(f"Found {len(sample_files)} sample files")
    

    for sample_file in sample_files:
   
        base_name = sample_file.replace('_samples.npy', '')
        label_file = base_name + '_labels.npy'
        
        sample_path = os.path.join(samples_dir, sample_file)
        label_path = os.path.join(labels_dir, label_file)
        

        if not os.path.exists(label_path):
            print(f"Warning: Label file not found for {sample_file}, skipping...")
            continue
        
        print(f"\nProcessing: {base_name}")
        

        dataset = GeneDataset(sample_path, label_path)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        print(f"  Total samples: {len(dataset)}")
        
    
        all_embeddings = []
        all_labels = []
        
        pbar = tqdm(dataloader, desc=f"  Generating embeddings")
        with torch.no_grad():
            for samples, labels in pbar:
           
                samples = samples.to(device).long()
                

                embeddings = model(samples)
           
                all_embeddings.append(embeddings.detach().cpu().numpy())
                all_labels.append(labels.numpy())
        

        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.concatenate(all_labels)
        
     
        embed_output_path = os.path.join(output_dir, f'{base_name}_embeds.npy')
        label_output_path = os.path.join(output_dir, f'{base_name}_labels.npy')
        
        np.save(embed_output_path, all_embeddings)
        np.save(label_output_path, all_labels)
        
        print(f"  Saved embeddings: {embed_output_path} (shape: {all_embeddings.shape})")
        print(f"  Saved labels: {label_output_path} (shape: {all_labels.shape})")
        print(f"  Label distribution: {np.bincount(all_labels.astype(int))}")


def load_paired_dataset(base_name, data_dir="./paired_data/"):

    embed_path = os.path.join(data_dir, f'{base_name}_embeds.npy')
    label_path = os.path.join(data_dir, f'{base_name}_labels.npy')
    
    embeddings = np.load(embed_path)
    labels = np.load(label_path)
    
    assert len(embeddings) == len(labels), "Embeddings and labels length mismatch"
    
    return embeddings, labels


class PairedEmbeddingDataset(Dataset):
  
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


if __name__ == "__main__":

    generate_embeddings_with_labels(
        samples_dir="./samples/",
        labels_dir="./labels/",
        output_dir="./paired_data/",
        pretrained_path="geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/",
        batch_size=32,  
        hidden_layer=6,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\n" + "="*60)
    print("Embedding generation completed!")
    print("="*60)
    

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
        
 
        paired_dataset = PairedEmbeddingDataset(embeddings, labels)
        paired_loader = DataLoader(paired_dataset, batch_size=16, shuffle=True)
        
        print(f"\nDataLoader created with {len(paired_dataset)} samples")
        

        for batch_embeds, batch_labels in paired_loader:
            print(f"  Batch embeddings shape: {batch_embeds.shape}")
            print(f"  Batch labels shape: {batch_labels.shape}")
            break

