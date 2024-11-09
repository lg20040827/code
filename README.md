# LSTM_Frame

## 1.project information 
1. title = {LSTM-FRAME:},
2. author = {Chu Pan, Tianwu Zhang, Guang Li},  
3. year = {2024},

## 2.Project profile
Doc2Vec was used to generate RNA sequence vector, Role2Vec was used to generate RNA expression profile vector, and LSTM model was used to predict ceRNA structure

## 3.Needed python package: 
- Pandas            1.4.3(For file read and write)
- Numpy             1.25.2(For file read and write)
- Gensim            4.2.0(For Doc2Vec)
- Networkx          2.8.8(For Role2vec)
- Keras             2.15.0(For LSTM)
- Sklearn           0.0(for comparison)
- Matplotlib        3.8.0(for roc curve and prc curve plot)

## 4.Code introduction

### 4.1.Doc2Vec
- 准备数据

  确保你有一个包含 RNA 序列的文本文件（RNASequence.txt)文件中的每一行应包含一个 mRNA 序列.

- 运行脚本

  python Doc2Vec.py

- 输出结果

  脚本会生成一个名为 SequenceVec.xlsx 的 Excel 文件，其中包含了每个 RNA 序列对应的向量表示。这些向量可以用于后续的分析或机器学习任务。

### 4.2.Pearson_correlation
- 准备数据

  确保你有一个包含癌症表达谱数据的 Excel 文件，例如 PRAD_miRNA_exp.xlsx，其中包含多维的基因或 miRNA 表达数据。

- 运行脚本

   python Pearson_correlation.py

- 输出结果

  运行脚本后，相关性矩阵将被计算并保存到指定路径（例如 PRAD_miRNA_network.csv）中。

### 4.3.Role2Vec
- 准备数据

  确保你有一个包含基因表达谱数据的 CSV 文件，其中包括基因节点的相关性矩阵。

- 运行脚本

  python  Role2Vec.py

- 输出结果

  脚本会生成一个 Excel 文件，其中包含每个基因节点对应的向量表示。你可以使用这些向量进行后续的分析、可视化或机器学习任务。

### 4.4.LSTM
- 准备数据

  准备好RNA的向量文件(将三种RNA的序列向量和表达谱向量拼接在一起),三种RNA的名称文件(每一行包含lncRNA,miRNA和mRNA的名称),以及每行数据的标签(能构成ceRNA结构的标签为1,反之为0)

- 运行脚本

  python LSTM.py

- 输出结果
  
  脚本会对模型进行评估，计算出AUC值并画出ROC曲线
