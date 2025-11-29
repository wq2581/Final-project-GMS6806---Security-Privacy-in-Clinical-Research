import csv
import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
from scipy import sparse
import anndata
import harmonypy as hm
import gzip

# 加载token字典
with open('token_dict.pkl', 'rb') as file:
    dictionary = pickle.load(file)

directory_path = "./rawdata/"
harmony_output_dir = "./harmony_processed/"
samples_output_dir = "./samples/"
labels_output_dir = "./labels/"

# 创建输出目录
os.makedirs(harmony_output_dir, exist_ok=True)
os.makedirs(samples_output_dir, exist_ok=True)
os.makedirs(labels_output_dir, exist_ok=True)


def harmony_preprocessing(directory_path, output_dir, n_hvgs=512):
    """
    对所有文件进行 Harmony 批次校正，只保留 top 512 HVGs
    """
    print("=" * 60)
    print("Step 1: Harmony Preprocessing")
    print("=" * 60)
    
    file_paths = [os.path.join(directory_path, f) 
                  for f in os.listdir(directory_path) 
                  if f.endswith(".txt") or f.endswith(".txt.gz")]
    
    if not file_paths:
        raise ValueError(f"No .txt or .txt.gz files found in {directory_path}")
    
    adatas = []
    
    # 读取所有文件
    for path in file_paths:
        print(f"Loading {path}...")
        
        if path.endswith('.gz'):
            df = pd.read_csv(path, sep="\t", index_col=0, compression='gzip')
        else:
            df = pd.read_csv(path, sep="\t", index_col=0)
        
        # 提取 Condition 列
        if "Condition" in df.columns:
            conditions = df["Condition"]
            df = df.drop(columns=["Condition"])
        else:
            conditions = pd.Series("NA", index=df.index)
        
        # 转换为稀疏矩阵
        X_sparse = sparse.csr_matrix(df.values)
        
        adata = sc.AnnData(X=X_sparse)
        adata.var_names = df.columns
        adata.obs_names = df.index
        adata.obs["batch"] = os.path.basename(path).replace(".txt.gz", "").replace(".txt", "")
        adata.obs["Condition"] = conditions.values
        
        adatas.append(adata)
    
    # 合并所有数据
    print("Merging all datasets...")
    adata_merged = anndata.concat(
        adatas,
        axis=0,
        join="outer",
        label="batch",
        keys=[a.obs["batch"][0] for a in adatas],
        fill_value=0
    )
    
    # 标准预处理流程
    print("Running preprocessing pipeline...")
    sc.pp.normalize_total(adata_merged, target_sum=1e4)
    sc.pp.log1p(adata_merged)
    
    # 选择高变基因 (HVGs)
    print(f"Selecting top {n_hvgs} highly variable genes...")
    sc.pp.highly_variable_genes(adata_merged, n_top_genes=n_hvgs, subset=True)
    
    sc.pp.scale(adata_merged, max_value=10)
    sc.tl.pca(adata_merged, n_comps=min(30, n_hvgs-1))
    
    # Harmony 批次校正
    print("Running Harmony batch correction...")
    ho = hm.run_harmony(adata_merged.obsm['X_pca'], adata_merged.obs, 'batch')
    adata_merged.obsm['X_pca_harmony'] = ho.Z_corr.T
    
    # 保存整体结果
    adata_merged.write(os.path.join(output_dir, "harmony_corrected_data.h5ad"))
    print(f"Saved harmony corrected data to {output_dir}/harmony_corrected_data.h5ad")
    
    # 提取表达矩阵
    expr_matrix = adata_merged.X.toarray() if sparse.issparse(adata_merged.X) else adata_merged.X
    df_corrected_all = pd.DataFrame(
        expr_matrix, 
        index=adata_merged.obs_names, 
        columns=adata_merged.var_names
    )
    
    # 拆分并保存每个 batch
    print("Splitting and saving individual batches...")
    for batch_name in adata_merged.obs["batch"].unique():
        cell_ids = adata_merged.obs.query("batch == @batch_name").index
        df_batch = df_corrected_all.loc[cell_ids].copy()
        
        # 添加 Condition 列到第一列
        conditions = adata_merged.obs.loc[cell_ids, "Condition"]
        df_batch.insert(0, "Condition", conditions.values)
        
        # 保存为压缩文件
        out_path = os.path.join(output_dir, f"{batch_name}.txt.gz")
        with gzip.open(out_path, 'wt') as f:
            df_batch.to_csv(f, sep='\t', float_format="%.4f")
        print(f"  Saved: {out_path}")
    
    # 返回HVG基因列表，用于后续token匹配
    hvg_genes = adata_merged.var_names.tolist()
    return hvg_genes


def dataset_generator(directory_path, hvg_genes, is_sorted=True, seq_length=512):
    """
    处理 Harmony 校正后的数据，生成样本和标签
    """
    print("\n" + "=" * 60)
    print("Step 2: Dataset Generation")
    print("=" * 60)
    
    csv.field_size_limit(500000000)
    files_list = [f for f in os.listdir(directory_path) if f.endswith('.txt.gz')]
    
    for filename in files_list:
        csv_file_path = os.path.join(directory_path, filename)
        print(f"\nProcessing {csv_file_path}...")
        
        with gzip.open(csv_file_path, 'rt', encoding='utf-8') as file:
            csv_reader = csv.reader(file, delimiter='\t')
            
            pattern = []
            labels = []
            samples = []
            headr = True
            
            for row in csv_reader:
                if headr:
                    # 处理表头，构建pattern
                    for gene in row:
                        gene = gene.replace('"', '').strip()
                        if gene in dictionary:
                            pattern.append(dictionary[gene])
                        else:
                            pattern.append(-99999)
                    headr = False
                else:
                    # 处理数据行
                    seq_pattern_order_id_EXPscore = []
                    
                    for i in range(len(row)):
                        if i == 0:
                            # 跳过第一列（cell ID）
                            pass
                        elif i == 1:
                            # 第二列是 Condition
                            if 'sensitive' in row[i].lower():
                                labels.append(1)
                            elif 'resistant' in row[i].lower():
                                labels.append(0)
                        else:
                            # 表达值
                            exp_value = row[i].strip()
                            if exp_value and exp_value != '0':
                                if pattern[i] != -99999:
                                    seq_pattern_order_id_EXPscore.append(
                                        (pattern[i], float(exp_value))
                                    )
                    
                    # 排序（按表达值降序）
                    if is_sorted:
                        seq_pattern_order_id_EXPscore = sorted(
                            seq_pattern_order_id_EXPscore, 
                            key=lambda x: x[1], 
                            reverse=True
                        )
                    
                    # 提取token ID
                    sample = [item[0] for item in seq_pattern_order_id_EXPscore]
                    
                    # Padding
                    while len(sample) < seq_length:
                        sample.append(0)
                    sample = sample[:seq_length]
                    
                    samples.append(sample)
        
        # 保存结果
        base_filename = filename.replace('.txt.gz', '')
        file_path_samples = os.path.join(samples_output_dir, f'{base_filename}_samples.npy')
        file_path_labels = os.path.join(labels_output_dir, f'{base_filename}_labels.npy')
        
        np.save(file_path_samples, np.array(samples))
        np.save(file_path_labels, np.array(labels))
        
        print(f"  Saved {len(samples)} samples and {len(labels)} labels")
        print(f"  Samples: {file_path_samples}")
        print(f"  Labels: {file_path_labels}")


if __name__ == "__main__":
    # Step 1: Harmony 预处理，选择 top 512 HVGs
    hvg_genes = harmony_preprocessing(
        directory_path=directory_path,
        output_dir=harmony_output_dir,
        n_hvgs=512
    )
    
    # Step 2: 处理 Harmony 校正后的数据
    dataset_generator(
        directory_path=harmony_output_dir,
        hvg_genes=hvg_genes,
        is_sorted=True,
        seq_length=512
    )
    
    print("\n" + "=" * 60)
    print("All processing completed!")
    print("=" * 60)