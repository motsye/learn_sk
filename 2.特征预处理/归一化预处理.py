# 通过对原始数据进行变换把数据映射到默认【0-1】之间
# 让所有的特征缩放 但容易收到异常点的影响
# 让所有特征权重相同 不让某一特征影响数据

from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler(feature_range=[0, 1])
data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
print(data)
