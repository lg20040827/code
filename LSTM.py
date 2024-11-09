import pandas as pd
import numpy as np
from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Model
from keras import regularizers
from keras.optimizers import Adam
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


combined_embeddings=pd.read_excel(r"E:\数据\其他癌症表达谱\BRCA\BRCA_Vec\BRCA_Train_data.xlsx")
len1=len(combined_embeddings)-1
len2=int(len1*0.5)
print(len2)

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
#绘制ROC曲线
def plot_roc_curve(auc, fpr, tpr):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='LSTM ROC curve (area = %.4f)' % auc)# 绘制ROC曲线，并标注AUC值
    plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
    plt.axis([0, 1, 0, 1])  # 设置坐标轴范围
    plt.xlabel('False Positive Rate', fontsize=16)  # 设置横轴标签
    plt.ylabel('True Positive Rate', fontsize=16)  # 设置纵轴标签
    plt.title('Receiver Operating Characteristic (ROC) Curve')  # 设置图标题
    plt.legend(loc="lower right")
plot_roc_curve(auc, fpr, tpr )
plt.show()
#输出top前十
# y_train_prop_array = np.array(y_test1_prob)
# prediction_df = pd.DataFrame({'Prediction Score': y_train_prop_array.ravel(), 'True Label': y_text1.ravel()})
# prediction_df['Original Index'] = prediction_df.index
# prediction_df_sorted = prediction_df.sort_values(by='Prediction Score',ascending=False)
# top_20_predictions = prediction_df_sorted.head(len1-len2)
# top_20_predictions=pd.DataFrame(top_20_predictions)
# top_20_predictions.to_excel("E:\数据\其他癌症表达谱\PRAD\PRAD_Vec\PRAD_Top1.xlsx")
# Top_20=[]
# for index in top_20_predictions['Original Index']:
#     Top_20.append(z_text[index])
# Top_20=pd.DataFrame(Top_20)
# Top_20.to_excel("E:\数据\其他癌症表达谱\PRAD\PRAD_Vec\PRAD_Top.xlsx")


from sklearn.metrics import precision_recall_curve, auc
# 计算精确率和召回率
precision, recall, _ = precision_recall_curve(y_text1.flatten(), y_test1_prob.flatten())
pr_auc = auc(recall, precision)

# 绘制 PR 曲线
# plt.figure()
# plt.plot(recall, precision, marker='.', color='blue')  # 这里可以设置你想要的颜色
# plt.title(f'PR Curve (AUC = {pr_auc:.4f})')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.grid()
# plt.show()
