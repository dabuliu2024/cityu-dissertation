
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 读取模拟数据 (假设文件名是sensor_data.csv)
data = pd.read_csv('sensor_data.csv')  # 假设数据文件在同一目录

# 假设数据有两列：time 和 voltage, 并且标签在 'label' 列
time = data['time'].values
voltage = data['voltage'].values
labels = data['label'].values

# 假设每个信号采样窗口为100个点，我们可以选择滑动窗口方法来处理信号数据
window_size = 100  # 每个时间窗口大小
step = 50  # 滑动窗口的步长
X = []
y = []

# 提取窗口特征：均值、标准差、最大值、最小值
for i in range(0, len(voltage) - window_size, step):
    window = voltage[i:i+window_size]
    
    # 计算窗口的统计特征
    mean = np.mean(window)
    std = np.std(window)
    max_val = np.max(window)
    min_val = np.min(window)
    
    X.append([mean, std, max_val, min_val])  # 这里可以添加更多的特征
    y.append(labels[i+window_size//2])  # 假设每个窗口的标签由中心点决定

X = np.array(X)
y = np.array(y)

# 切分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 训练支持向量机
clf = SVC(kernel='linear')  # 使用线性核
clf.fit(X_train, y_train)

# 预测与评估
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 结果可视化
plt.plot(time, voltage, label='Voltage Signal')
plt.title('TENG Sensor Voltage Signal')
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.legend()
plt.show()
