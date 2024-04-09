from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import pandas as pd


def mean_calc_rf(model, train_X, val_X, train_y, val_y):
    model.fit(train_X,train_y)
    prediction = model.predict(val_X)
    mae = mean_absolute_error(val_y, prediction)
    return mae


def ordinal_encoding(X):
    X['Sex'] = X['Sex'].where(X.Sex == 'male',0)
    X['Sex'] = X['Sex'].where(X.Sex != 'male',1)
    return X

#work with missing values
#drop columns with missing values 
def drop_columns_missing_values(data):
    col_with_missing = [col for col in data.keys() if data[col].isnull().any()]
    
    return data.drop(col_with_missing, axis=1)

#replace missign value (Imputation)
def imputation(data):
    my_imputer = SimpleImputer()
    
    new_data = pd.DataFrame(my_imputer.fit_transform(data))
    new_data.columns = data.columns
    return new_data

#read data
X_full = pd.read_csv('data/train.csv')
X_test_full = pd.read_csv('data/test.csv')

#remove row with missing target
X_full.dropna(axis=0, subset='Survived', inplace=True)
y = X_full.Survived
X_full.drop(['Survived'], axis=1, inplace=True)

#Ordinal Encoding 'Sex' 
X_full = ordinal_encoding(X_full)
X_test_full = ordinal_encoding(X_test_full)

#One-hot Encodign
# 

#Only use numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

#split data for better validation
train_X, val_X, train_y, val_y = train_test_split(X_full, y, random_state=1, train_size=0.8,test_size=0.2)

#deal with missing values
train_X = imputation(train_X)
val_X = imputation(val_X)


models = [RandomForestRegressor(n_estimators=100, random_state=1),
          RandomForestRegressor(n_estimators=50, random_state=1),
          RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=1),
          RandomForestRegressor(n_estimators= 200, min_samples_split=20, random_state=0),
          RandomForestRegressor(n_estimators= 100, max_depth=8, random_state=0)
        ]

best = -1
model = None

#find best model
for m in models:
    temp = mean_calc_rf(m,train_X,val_X,train_y,val_y)
    if best==-1 or best > temp:
        best = temp
        model = m

#train the model
X = imputation(X)
model.fit(X,y)

test_predict = model.predict(X_test)

output = pd.DataFrame({'PassengerId': X_test_full.PassengerId,
                       'Survived': test_predict
                      })
output.to_csv('submission_1.csv', index=False)