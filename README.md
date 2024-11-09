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
- Prepare data

  Make sure you have a text file that contains the RNA sequence (RNASequence.txt) Each line in the file should contain an mRNA sequence.

- Run script

  python Doc2Vec.py

- Output result

  The script generates an Excel file called SequenceVec.xlsx that contains the vector representation for each RNA sequence. These vectors can be used for subsequent analysis or machine 
  learning tasks.

### 4.2.Pearson_correlation
- Prepare data

  Make sure you have an Excel file that contains cancer expression profile data, such as PRAD_miRNA_exp.xlsx, with multidimensional gene or miRNA expression data.

- Run the script

   python Pearson_correlation.py

- Output result

  After running the script, the correlation matrix is calculated and saved to the specified path (for example, PRAD_miRNA_network.csv).

### 4.3.Role2Vec
- Prepare data

  Make sure you have a CSV file that contains the gene expression profile data, including the correlation matrix of the gene nodes.

- Run script

  python  Role2Vec.py

- Output result

  The script generates an Excel file with a vector representation for each gene node. You can use these vectors for subsequent analysis, visualization, or machine learning tasks.

### 4.4.LSTM
- Prepare data

  Prepare the vector file of RNA (splicing the sequence vector and expression profile vector of the three Rnas together), the name file of the three Rnas (each line contains the names of 
  lncRNA,miRNA and mRNA), and the label of each line of data (the label that can form the structure of ceRNA is 1, and the label of the other is 0).

- Run script

  python LSTM.py

- Output result
  
  The script evaluates the model, calculates the AUC value and draws the ROC curve
