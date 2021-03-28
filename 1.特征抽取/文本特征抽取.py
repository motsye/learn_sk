from sklearn.feature_extraction.text import CountVectorizer
import jieba

mydata_en = ['life is short', 'i love python', 'life is too long']
mydata_zh = [' '.join(i) for i in list(map(lambda x: list(jieba.cut(x)), ['人生苦短', '我爱python', '人生漫长']))]

cv = CountVectorizer()

res = cv.fit_transform(mydata_zh)

# 实质是统计所有词在每篇文本中的出现 出现为N次 不出现为0
print(res.toarray())

# 统计文章中所有的词 对单个英文字母或汉字不统计
print(cv.get_feature_names())
