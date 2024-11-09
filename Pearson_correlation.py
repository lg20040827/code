import pandas as pd
import numpy as np

# 加载数据
data = pd.read_excel(r"E:\数据\其他癌症表达谱\PRAD\PRAD_correct\PRAD_miRNA_exp.xlsx")

# 选择数据的子集，并转置数据
data_values = data.iloc[:, 2:].values
data_values_transposed = data_values.T

# 将转置后的 numpy.ndarray 转换为 pandas DataFrame
df = pd.DataFrame(data_values_transposed)

# 计算皮尔森相关系数矩阵
correlation_matrix = df.corr(method='pearson')

correlation_matrix.to_csv(r"E:\数据\其他癌症表达谱\PRAD\PRAD_Vec\PRAD_miRNA_network.csv")
# # 使用 xlsxwriter 保存结果到 Excel
# with pd.ExcelWriter(r"E:\数据\其他癌症表达谱\BRCA\BRCA_lncRNA_network.xls", engine='xlwt') as writer:
#     correlation_matrix.to_excel(writer, sheet_name='Correlation Matrix')
