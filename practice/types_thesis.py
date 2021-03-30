import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def foo():
    thesis_data = pd.read_csv('../data/thesis_data.csv')
    thesis_data = thesis_data.dropna(axis=0, how='any')

    # 获取特征值和目标值
    features_values = [re.sub(r':|\r|\n|\t', '', i) for i in thesis_data['Keywords']]

    targets_values = [re.sub(r':|\r|\t|\n', '', i) for i in thesis_data['Categories']]

    # 分割训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(features_values, targets_values, test_size=0.25)

    # 进行特征工程
    cv = TfidfVectorizer()
    x_train = cv.fit_transform(x_train)
    x_test = cv.transform(x_test)

    # 使用贝叶斯预测
    mt = MultinomialNB()
    mt.fit(x_train, y_train)

    # 测试集的类别
    y_predict = mt.predict(x_test)
    print(y_predict)

    # 正确率
    print(mt.score(x_test, y_test))

    # 召回率
    print(classification_report(y_test, y_predict))


foo()
