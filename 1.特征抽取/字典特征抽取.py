from sklearn.feature_extraction import DictVectorizer

mydata = [{'city': '北京', 'code': '100'}, {'city': '杭州', 'code': '2'}, {'city': '上海', 'code': 20}]

dic = DictVectorizer(sparse=False)

# 抽取特征值
res = dic.fit_transform(mydata)
print(res)

# 查看数据特征值
res = dic.inverse_transform(res)

# 特征名称
print(dic.get_feature_names())
