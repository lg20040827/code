import pandas as pd
import numpy as np
from scipy.stats import pearsonr, linregress

database = pd.read_csv(r"E:\数据\其他癌症表达谱\LIHC\LIHC_precission\LIHC_demo1.csv")
lnc_exp = pd.read_excel(r"E:\数据\其他癌症表达谱\LIHC\LIHC_correct\LIHC_lncRNA_exp.xlsx")
m_exp = pd.read_excel(r"E:\数据\其他癌症表达谱\LIHC\LIHC_correct\LIHC_mRNA_exp.xlsx")


# 读取三种RNA名称列表
with open(r"E:\数据\其他癌症表达谱\LIHC\LIHC_mRNA_name.txt", 'r') as file:
    mRNA_name = file.read().splitlines()

with open(r"E:\数据\其他癌症表达谱\LIHC\LIHC_miRNA_name.txt", 'r') as file:
    miRNA_name = file.read().splitlines()

with open(r"E:\数据\其他癌症表达谱\LIHC\LIHC_lncRNA_name.txt", 'r') as file:
    lncRNA_name = file.read().splitlines()

data=[]
for i in range(0,len(database)):
    x = database.iloc[i, database.columns.get_loc('mRNA')]
    y = database.iloc[i, database.columns.get_loc('miRNA')]
    z = database.iloc[i, database.columns.get_loc('lncRNA')]
    mRNA_index=-1
    lncRNA_index=-1
    miRNA_index=-1
    for index, line in enumerate(mRNA_name):
        if x == line:
            mRNA_index = index
            break

    for index, line in enumerate(miRNA_name):
        if y == line:
            miRNA_index = index
            break

    for index, line in enumerate(lncRNA_name):
        if z == line:
            lncRNA_index = index
            break
    # 提取对应基因的数据（注意转换为浮点数类型）

    x_exp = m_exp.iloc[mRNA_index, 2:].astype(float).dropna().values
    y_exp = lnc_exp.iloc[lncRNA_index, 2:].astype(float).dropna().values
    # print(x_exp)
    # print(y_exp)
    pearson_corr, _ = pearsonr(x_exp, y_exp)
    if pearson_corr<-0.3:
        print(x,z)
        data.append([x, y, z, pearson_corr])

df = pd.DataFrame(data, columns=['mRNA', 'miRNA', 'lncRNA', 'pearsonr'])
df.to_excel("E:\数据\其他癌症表达谱\LIHC\LIHC_vec\LIHC_fyb.xlsx")