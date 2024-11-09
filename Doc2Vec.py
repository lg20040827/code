import gensim
import pandas as pd
from gensim.models.doc2vec import Doc2Vec

text_list = []  # 初始化一个空列表

with open("E:\数据\最终数据\序列数据\mRNA\mRNASequence.txt",'r') as file:
    for line in file:
        text=line.split(":")[-1].strip()
        #text_list.append(line.scrip())
        text_list.append(text)
# 输出列表内容
print(text_list)

new_text_list = []

for line in text_list:
    fragments = [line[i:i+8] for i in range(0, len(line), 8)]
    new_line = ' '.join(fragments)
    new_text_list.append(new_line)
'''
for line in text_list:
    fragments = [line[i:i+8] for i in range(len(line)-7)]
    new_line = ' '.join(fragments)
    new_text_list.append(new_line)
'''
print(new_text_list)
def getText():
    df_train = pd.DataFrame(new_text_list, columns=['Text'])
    return df_train

text_df = getText()

TaggededDocument=gensim.models.doc2vec.TaggedDocument
def preprocess_text(text_df):
    tagged_data = []
    for index, row in text_df.iterrows():
        tokens = row['Text'].split()
        tagged_data.append(TaggededDocument(words=tokens, tags=[index]))
    return tagged_data
c=preprocess_text(text_df)

print(c[0])

def train(c,size=128):
    model = Doc2Vec(c, dm=1, min_count=15, window=5, vector_size=size, sample=0, negative=5, workers=5)
    model.train(c, total_examples=model.corpus_count, epochs=50)
    return model

model_dm=train(c)



vectors = [model_dm.dv[i] for i in range(len(text_df))]
df=pd.DataFrame(vectors)
df.to_excel("E:\数据\最终数据\序列数据\mSequenceVec.xlsx")


