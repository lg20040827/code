import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# 生成图结构
def create_graph(correlation_matrix, threshold):
    num_nodes = correlation_matrix.shape[0]
    graph = nx.Graph()

    for i in range(num_nodes):
        graph.add_node(i)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if correlation_matrix[i, j] >= threshold:
                graph.add_edge(i, j)

    return graph


# 使用 Role2Vec 训练节点向量
def role2vec(graph, dimensions, walk_length, num_walks, window_size):
    walks = generate_walks(graph, num_walks, walk_length)
    model = Word2Vec(walks, vector_size=dimensions, window=window_size, min_count=1, sg=0, workers=4)
    return model.wv


# 生成节点序列
def generate_walks(graph, num_walks, walk_length):
    walks = []

    for _ in range(num_walks):
        for node in graph.nodes():
            walk = random_walk(graph, walk_length, start_node=node)
            walks.append(walk)

    return walks


# 随机游走生成节点序列
def random_walk(graph, walk_length, start_node):
    walk = [start_node]

    for _ in range(walk_length - 1):
        neighbors = list(graph.neighbors(walk[-1]))
        if len(neighbors) > 0:
            walk.append(np.random.choice(neighbors))
        else:
            break

    return walk


# 示例相关性矩阵
df = pd.read_csv(r"E:\数据\其他癌症表达谱\PRAD\PRAD_Vec\PRAD_miRNA_network.csv")
data1 = df.iloc[:, 1:].values
correlation_matrix = np.array(data1)

# 设置阈值和其他参数
threshold = 0.4
dimensions = 128
walk_length = 50
num_walks = 20
window_size = 10

# 创建图结构
graph = create_graph(correlation_matrix, threshold)
print(graph)

# 使用 Role2Vec 获取节点向量
node_vectors = role2vec(graph, dimensions, walk_length, num_walks, window_size)
print(node_vectors)

# 获取单个节点的向量
# 遍历图中的每个节点，输出其向量表示
vector_list = []
for node in graph.nodes():
    vector = node_vectors[node]
    vector_list.append(vector)

df1 = pd.DataFrame(vector_list)
df1.to_excel("E:\数据\其他癌症表达谱\PRAD\PRAD_Vec\PRAD_miRNA_ExpVec.xlsx")



