from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd


def mean_calc_rf(max_leaf, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes = max_leaf, random_state=1)
    model.fit(train_X,train_y)
    prediction = model.predict(val_X)
    mae = mean_absolute_error(val_y,prediction)
    return mae

train_path = 'data/train.csv'
data_train = pd.read_csv(train_path)

y = data_train['Survived']
features = ['Age', 'Pclass', 'Sex', 'Cabin', 'Parch']
X = data_train[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

error = -1
size = -1
for i in range(1,31):
    temp = mean_calc_rf(i*i, train_X, val_X, train_y, val_y)
    if error == -1 or error > temp:
        error = temp
        size = i*i


rf_model = RandomForestRegressor(max_leaf_nodes = size, random_state=1)
rf_model.fit(X,y)

test_path = 'data/test.csv'
data_test = pd.read_csv(test_path)

test_X = data_test[features]
test_predict = rf_model.predict(text_X)

output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Survived': test_predict
                      })
output.to_csv('submission_1.csv', index=False)