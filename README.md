import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.metrics import accuracy_score
import seaborn as sns

# 加载数据
data = pd.read_csv('sensor_data.csv')

# 将标签转换为数值
data['label'] = data['label'].map({'walking': 0, 'running': 1})

# 设置时间窗口（每100个时间步作为一个样本）
window_size = 100
X_data = []
y_data = []

# 以100步为一个窗口，构建样本数据
for i in range(0, len(data) - window_size, window_size):
    X_data.append(data['voltage'].iloc[i:i + window_size].values)
    y_data.append(data['label'].iloc[i])

# 转换为numpy数组
X_data = np.array(X_data)
y_data = np.array(y_data)

# 扩展X_data的维度，使其符合LSTM输入格式 (样本数, 时间步长, 特征数)
X_data = np.expand_dims(X_data, axis=-1)  # 添加特征维度

# 数据标准化
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data.reshape(-1, 1)).reshape(X_data.shape)

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

# 创建LSTM模型
model = Sequential()

# 添加双向LSTM层
model.add(Bidirectional(LSTM(units=64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))

# 添加Dropout层防止过拟合
model.add(Dropout(0.3))

# 添加第二个LSTM层
model.add(Bidirectional(LSTM(units=64)))

# 添加Dropout层
model.add(Dropout(0.3))

# 添加全连接层
model.add(Dense(units=1, activation='sigmoid'))  # 二分类问题使用sigmoid激活函数

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 可视化训练和验证损失和准确率
plt.figure(figsize=(12, 6))

# 损失图
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 准确率图
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 评估模型
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)  # 阈值为0.5，输出大于0.5则认为是步态2，反之步态1

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 可视化混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Gait 1', 'Gait 2'], yticklabels=['Gait 1', 'Gait 2'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
