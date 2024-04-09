import pandas as pd


train_path = 'data/train.csv'
data_train = pd.read_csv(train_path)


features = ['Age', 'Pclass', 'Sex', 'SibSp', 'Parch']
data_train['Sex'] = data_train['Sex'].where(data_train.Sex=='male',0)
data_train['Sex'] = data_train['Sex'].where(data_train.Sex!='male',1)

data_train = data_train[features].dropna(how='any')

print(data_train[features])