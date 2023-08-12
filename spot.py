import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier

lab=LabelEncoder()
data=pd.read_excel("Spotify_data.xlsx")
print(data.columns)
print(data.describe())
print(data.info())
print(data.isna().sum())
print('------------The number columns---------------')
print(data.select_dtypes(include='number').columns.values)
print('--------The categorical columns-----------------')
print(data.select_dtypes(include='object').columns.values)

'''plt.figure(figsize=(17, 6))
corr = data.corr(method='spearman')
my_m = np.triu(corr)
sn.heatmap(corr, mask=my_m, annot=True, cmap="Set2")
plt.show()

for i in data.columns.values:
    if len(data[i].value_counts()) <=5 :
        print(i)
        print(data[i].value_counts().index)

for i in data.columns.values:
    if len(data[i].value_counts()) <= 5:
        for j in data[i].value_counts().index.values:
            print('------------------------------------------')
            print(f"The information about the column {i}")
            val=data[data[i]==j]
            for k in val.select_dtypes(include='object').columns.values:
                index = val[k].value_counts().index.values
                value = val[k].value_counts().values
                if (len(index) and len(value)) <= 10:
                    plt.pie(value, labels=index, autopct='%1.1f%%')
                    plt.title(f'the values and their counts related to songs  on spotify for  {i} column')
                    plt.legend()
                    plt.show()


satisfaction=data['pod_variety_satisfaction'].value_counts().index
pref_pod=data['preffered_pod_duration'].value_counts().index'''
for i in data.select_dtypes(include='object').columns.values:
    data[i]=lab.fit_transform(data[i])


'''for i in data.select_dtypes(include='number').columns.values:
    for j in data.select_dtypes(include='number').columns.values:
        sn.distplot(data[i], label=f"{i}", color='red')
        sn.distplot(data[j], label=f"{j}", color="blue")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()

for i in data.select_dtypes(include='number').columns.values:
    for j in data.select_dtypes(include='number').columns.values:
        sn.histplot(data[i], label=f"{i}", color='red')
        sn.histplot(data[j], label=f"{j}", color="blue")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()'''

y=[]
for i in data.columns.values:
    if len(data[i].value_counts()) <=3:
        y.append(i)


for i in y:
    x=data.drop(i,axis='columns')
    y=data[i]
    print('------------------------------------')
    print('-------------------------------------------------')
    print(f'The Dependent Variables are {x.columns.values}')
    print(f' prediction regarding {i.upper()} column')
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    print(y_train.shape)

    lr = LogisticRegression(max_iter=200)
    lr.fit(x_train, y_train)
    print('The logistic regression: ', lr.score(x_test, y_test))

    xgb = XGBClassifier()
    xgb.fit(x_train, y_train)
    print("the Xgb : ", xgb.score(x_test, y_test))

    lgb = LGBMClassifier()
    lgb.fit(x_train, y_train)
    print('The LGB', lgb.score(x_test, y_test))

    tree = DecisionTreeClassifier(criterion='gini', max_depth=1)
    tree.fit(x_train, y_train)
    print('Dtree ', tree.score(x_test, y_test))

    rforest = RandomForestClassifier(criterion='gini')
    rforest.fit(x_train, y_train)
    print('The random forest: ', rforest.score(x_test, y_test))

    adb = AdaBoostClassifier()
    adb.fit(x_train, y_train)
    print('the adb ', adb.score(x_test, y_test))

    grb = GradientBoostingClassifier()
    grb.fit(x_train, y_train)
    print('Gradient boosting ', grb.score(x_test, y_test))

    bag = BaggingClassifier()
    bag.fit(x_train, y_train)
    print('Bagging', bag.score(x_test, y_test))