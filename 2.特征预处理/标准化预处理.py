# 不容易收到异常点的影响 对数据进行缩放
# 方差 每一列数据波动的大小或者离散大小

from sklearn.preprocessing import StandardScaler

std = StandardScaler()
data = std.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
print(data)
