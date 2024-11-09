import pandas as pd
import random
import numpy as np
from keras.src.layers import MultiHeadAttention
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, concatenate, Dropout
from keras.models import Model
from keras import regularizers
import warnings
from keras.optimizers import Adam
import os
from sklearn.metrics import roc_curve, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from keras.callbacks import ReduceLROnPlateau


combined_embeddings=pd.read_excel(r"E:\数据\其他癌症表达谱\BRCA\BRCA_Vec\BRCA_Train_data.xlsx")
len1=len(combined_embeddings)-1
len2=int(len1*0.5)
combined_embeddings=combined_embeddings.iloc[1:, 1:].values[:len1]
Name=pd.read_excel(r"E:\数据\其他癌症表达谱\BRCA\BRCA_Vec\BRCA_Train_name.xlsx")

lab = pd.read_excel(r"E:\数据\其他癌症表达谱\BRCA\BRCA_Vec\BRCA_Train_lables.xlsx")
values_to_assign = lab.iloc[0:,1:].values[:len1]
Name=Name.iloc[0:,1:].values[:len1]
np.random.seed(42)
shuffle_index = np.random.permutation(len1)
values_to_assign,combined_embeddings,Name= values_to_assign[shuffle_index],combined_embeddings[shuffle_index],Name[shuffle_index]
values_to_assign,y_text,combined_embeddings,x_text,Name,z_text=values_to_assign[:len2],values_to_assign[len2:],combined_embeddings[:len2],combined_embeddings[len2:],Name[:len2],Name[len2:]
combined_embeddings=np.array(combined_embeddings)

from sklearn.preprocessing import MinMaxScaler
# 创建 Min-Max 归一化器
scaler = MinMaxScaler()
# 对数据进行 Min-Max 归一化
combined_embeddings = scaler.fit_transform(combined_embeddings)

l=len1
cs=0.0001
combined_embeddings_reshaped = combined_embeddings.reshape((len2, 1, 768))
x_test=x_text.reshape(len1-len2,1,768)

print("combined_embeddings_reshaped的形状:", combined_embeddings_reshaped.shape)
# 定义输入层
input_data = Input(shape=(combined_embeddings_reshaped.shape[1], combined_embeddings_reshaped.shape[2]),
                       name='input_data')

# 定义第一个LSTM层
lstm_output_1 = LSTM(units=64, activation='tanh', return_sequences=True,
                     kernel_regularizer=regularizers.l2(cs),
                     recurrent_regularizer=regularizers.l2(cs),
                     bias_regularizer=regularizers.l2(cs))(input_data)
lstm_output_1 = Dropout(0.01)(lstm_output_1)

# 定义第二个LSTM层
lstm_output_2 = LSTM(units=64, activation='tanh', return_sequences=True,
                     kernel_regularizer=regularizers.l2(cs),
                     recurrent_regularizer=regularizers.l2(cs),
                     bias_regularizer=regularizers.l2(cs))(lstm_output_1)
lstm_output_2 = Dropout(0.01)(lstm_output_2)

# 定义输出层
output = Dense(units=1, activation='sigmoid')(lstm_output_2)


num_samples = len2
# 将数据赋值给 labels 数组
labels = np.zeros(num_samples)
labels = values_to_assign
# 将 labels 重塑为 (num_samples, 1, 1) 的形状
labels = labels.reshape((num_samples, 1, 1))
lables1=np.zeros(len1-len2)
lables1=y_text
y_text1= lables1.reshape((len1-len2, 1, 1))

X=combined_embeddings_reshaped
y=labels

# 定义模型
model = Model(inputs=input_data, outputs=output)
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.00001), metrics=['accuracy'])

model.fit(X, y, batch_size=30, epochs=400,validation_data=(x_test, y_text1))

y_test1_prob = model.predict(x_test)
fpr, tpr, thresholds = roc_curve(y_text1.flatten(), y_test1_prob.flatten())
auc = roc_auc_score(y_text1.flatten(), y_test1_prob.flatten())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

combined_embeddings=pd.read_excel("E:\数据\其他癌症表达谱\BRCA\BRCA_Vec\BRCA_Train_data.xlsx")

Name=pd.read_excel(r"E:\数据\其他癌症表达谱\BRCA\BRCA_Vec\BRCA_Train_name.xlsx")
len=len(combined_embeddings)-1
len1=int(len1*0.5)
combined_embeddings=combined_embeddings.iloc[1:, 1:].values[:len]
combined_embeddings=np.array(combined_embeddings)

from sklearn.preprocessing import MinMaxScaler
# 创建 Min-Max 归一化器
scaler = MinMaxScaler()
# 对数据进行 Min-Max 归一化
X = scaler.fit_transform(combined_embeddings)
# 读取 Excel 文件
y = pd.read_excel(r"E:\数据\其他癌症表达谱\BRCA\BRCA_Vec\BRCA_Train_lables.xlsx")

#洗牌操作
y = y.iloc[0:,1:].values[:len]
Name=Name.iloc[0:,1:].values[:len]
np.random.seed(42)
shuffle_index = np.random.permutation(len)
X,y,Name = X[shuffle_index],y[shuffle_index],Name[shuffle_index]
#y  = y[shuffle_index]
y = y.reshape(-1)

X_train,X_text,y_train,y_text,Name_train,Name_text= X[:len1],X[len1:],y[:len1],y[len1:],Name[:len1],Name[len1:]

y_train=(y_train==1)

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(max_iter=100,random_state=42)

from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf,X_train,y_train,cv=5,scoring='accuracy')
print(cross_val_score(sgd_clf,X_train,y_train,cv=5,scoring='accuracy'))

from sklearn.model_selection import StratifiedGroupKFold
skflods = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)

#五折交叉验证
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf,X_train,y_train,cv=5)
print(confusion_matrix(y_train,y_train_pred))
y_scores =  cross_val_predict(sgd_clf,X_train,y_train,cv=5,method="decision_function")

from sklearn.metrics import precision_recall_curve
precisions,recalls,thresholds = precision_recall_curve(y_train,y_scores)

from sklearn.metrics import roc_curve
fpr1,tpr1,thresholds1=roc_curve(y_train,y_scores,drop_intermediate=False)


def plot_roc_curve(auc, fpr, tpr,auc1,fpr1,tpr1, label=None):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr1, tpr1, color='darkorange', lw=2, label='SGD ROC curve (area = %.4f)' % auc1)
    plt.plot(fpr, tpr, color='blue', lw=2, label='LSTM ROC curve (area = %.4f)' % auc)# 绘制ROC曲线，并标注AUC值
    plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
    plt.axis([0, 1, 0, 1])  # 设置坐标轴范围
    plt.xlabel('False Positive Rate', fontsize=16)  # 设置横轴标签
    plt.ylabel('True Positive Rate', fontsize=16)  # 设置纵轴标签
    plt.title('Receiver Operating Characteristic (ROC) Curve')  # 设置图标题
    plt.legend(loc="lower right")

from sklearn.metrics import roc_auc_score
sgd_clf.fit(X_train,y_train)
y_pred = sgd_clf.predict(X_text)
y_text_pred= sgd_clf.decision_function(X_text)
fpr,tpr,thresholds=roc_curve(y_text,y_text_pred,drop_intermediate=False)

auc1 = roc_auc_score(y_text, y_text_pred)
plot_roc_curve(auc,fpr,tpr,auc1,fpr1,tpr1)
print("AUC:", auc1)
plt.show()

