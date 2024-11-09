import pandas as pd
import numpy as np

# 读取三种RNA名称列表
with open(r"E:\数据\最终数据\序列数据\mRNA\mSequenceName.txt", 'r') as file:
    mRNA_name = file.read().splitlines()

with open(r"E:\数据\ceRNA数据\miname.txt", 'r') as file:
    miRNA_name = file.read().splitlines()

with open(r"E:\数据\最终数据\序列数据\lncRNA\lncGene.txt", 'r') as file:
    lncRNA_name = file.read().splitlines()

misequence_data=pd.read_excel(r"E:\数据\最终数据\序列数据\miSequenceVec.xlsx")
lncsequence_data=pd.read_excel(r"E:\数据\最终数据\序列数据\lncSequenceVec.xlsx")
msequence_data=pd.read_excel(r"E:\数据\最终数据\序列数据\mSequenceVec.xlsx")

miexp_name=pd.read_excel(r"E:\数据\其他癌症表达谱\PRAD\PRAD_miRNA_name.xlsx")
lncexp_name=pd.read_excel(r"E:\数据\其他癌症表达谱\PRAD\PRAD_lncRNA_name.xlsx")
mexp_name=pd.read_excel(r"E:\数据\其他癌症表达谱\PRAD\PRAD_mRNA_name.xlsx")

mexp_data=pd.read_csv(r"E:\数据\其他癌症表达谱\PRAD\PRAD_Vec\PRAD_mRNA_ExpVec.csv")
lncexp_data=pd.read_excel(r"E:\数据\其他癌症表达谱\PRAD\PRAD_Vec\PRAD_lncRNA_ExpVec.xlsx")
miexp_data=pd.read_excel(r"E:\数据\其他癌症表达谱\PRAD\PRAD_Vec\PRAD_miRNA_ExpVec.xlsx")
moutput_data=[]
mname=[]
mioutput_data=[]
miname=[]
lncoutput_data=[]
lncname=[]
for i in range(0,len(mexp_name)):
    x = mexp_name.iloc[i, mexp_name.columns.get_loc('mRNA')]
    print(x)
    mRNA_exp_vec = mexp_data.iloc[i, 1:].values
    mRNA_index = -1
    for index, line in enumerate(mRNA_name):
        if x == line:
            mRNA_index = index
            break
    if mRNA_index != -1 :
        mname.append(x)
        mRNA_seq_vec = msequence_data.iloc[mRNA_index, 1:].values
        merged_vector = np.concatenate([mRNA_seq_vec, mRNA_exp_vec], axis=0)
        moutput_data.append(merged_vector)

for i in range(0,len(miexp_name)):
    x = miexp_name.iloc[i, miexp_name.columns.get_loc('miRNA')]
    print(x)
    miRNA_exp_vec = miexp_data.iloc[i, 1:].values
    miRNA_index = -1
    for index, line in enumerate(miRNA_name):
        if x == line:
            miRNA_index = index
            break
    if miRNA_index != -1 :
        miname.append(x)
        miRNA_seq_vec = misequence_data.iloc[miRNA_index, 1:].values
        merged_vector = np.concatenate([miRNA_seq_vec, miRNA_exp_vec], axis=0)
        mioutput_data.append(merged_vector)

for i in range(0,len(lncexp_name)):
    x = lncexp_name.iloc[i, lncexp_name.columns.get_loc('lncRNA')]
    print(x)
    lncRNA_exp_vec = lncexp_data.iloc[i, 1:].values
    lncRNA_index = -1
    for index, line in enumerate(lncRNA_name):
        if x == line:
            lncRNA_index = index
            break
    if lncRNA_index != -1 :
        lncname.append(x)
        lncRNA_seq_vec = lncsequence_data.iloc[lncRNA_index, 1:].values
        merged_vector = np.concatenate([lncRNA_seq_vec, lncRNA_exp_vec], axis=0)
        lncoutput_data.append(merged_vector)


mname_df=pd.DataFrame(mname)
moutput_df=pd.DataFrame(moutput_data)
moutput_df.to_excel(r"E:\数据\其他癌症表达谱\PRAD\PRAD_Vec\PRAD_mvec.xlsx")
mname_df.to_excel(r"E:\数据\其他癌症表达谱\PRAD\PRAD_Vec\PRAD_mname.xlsx")

miname_df=pd.DataFrame(miname)
mioutput_df=pd.DataFrame(mioutput_data)
mioutput_df.to_excel(r"E:\数据\其他癌症表达谱\PRAD\PRAD_Vec\PRAD_mivec.xlsx")
miname_df.to_excel(r"E:\数据\其他癌症表达谱\PRAD\PRAD_Vec\PRAD_miname.xlsx")

lncname_df=pd.DataFrame(lncname)
lncoutput_df=pd.DataFrame(lncoutput_data)
lncoutput_df.to_excel(r"E:\数据\其他癌症表达谱\PRAD\PRAD_Vec\PRAD_lncvec.xlsx")
lncname_df.to_excel(r"E:\数据\其他癌症表达谱\PRAD\PRAD_Vec\PRAD_lncname.xlsx")