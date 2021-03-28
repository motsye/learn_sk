# 通过平均数或者中位数等对缺失值进行填补


from sklearn.impute import SimpleImputer
import numpy as np

si = SimpleImputer(missing_values=np.nan, strategy='mean')

data = si.fit_transform([[1, 2], [np.nan, 4], [5, 6]])

print(data)
