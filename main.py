from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


def mean_calc_rf(model, train_X, val_X, train_y, val_y):
    model.fit(train_X,train_y)
    prediction = model.predict(val_X)
    mae = mean_absolute_error(val_y, prediction)
    return mae

#Categorical Values
#Ordinal Encoding
def ordinal_encoding(X_train, X_valid):
    objects_col = [col for col in X_train.columns if X_train[col].dtype == 'object']
    good_col = [col for col in objects_col if set(X_valid[col]).issubset(X_train[col])]
    bad_col = list(set(objects_col) - set(good_col))
    
    new_X_train = X_train.drop(bad_col, axis=1)
    new_X_valid = X_valid.drop(bad_col, axis=1)
    ordinal_enc = OrdinalEncoder()
    
    new_X_train[good_col] = ordinal_enc.fit_transform(new_X_train[good_col])
    new_X_valid[good_col] = ordinal_enc.fit_transform(new_X_valid[good_col])
    return new_X_train, new_X_valid


#One-Hot Encoding
def oneHot_encoding(X_train, X_valid):
    objects_col = [col for col in X_train.columns if X_train[col].dtype == 'object']
    low_cardinal_cols = [col for col in objects_col if X_train[col].nunique() < 10]
    high_cadinal_cols = list(set(objects_col) - set(low_cardinal_cols))
    
    oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    oh_X_t = pd.DataFrame(oh_encoder.fit_transform(X_train[low_cardinal_cols]))
    oh_X_v = pd.DataFrame(oh_encoder.transform(X_valid[low_cardinal_cols]))
    oh_X_t.index = X_train.index
    oh_X_v.index = X_valid.index
    
    X_train = X_train.drop(objects_col, axis=1)
    X_valid = X_valid.drop(objects_col, axis=1)
    X_train = pd.concat([X_train, oh_X_t], axis=1)
    X_valid = pd.concat([X_valid, oh_X_v], axis=1)
    
    X_train.columns = X_train.columns.astype(str)
    X_valid.columns = X_valid.columns.astype(str)
    return X_train, X_valid   

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

X_full = drop_columns_missing_values(X_full)
X_test_full = drop_columns_missing_values(X_test_full)
#Ordinal Encoding
X , X_test= ordinal_encoding(X_full, X_test_full)

# #One-hot Encodign
# # 
# X = oneHot_encoding(X_full)
# X_test = oneHot_encoding(X_test_full)

# #Only use numerical predictors
# X = X_full.select_dtypes(exclude=['object'])
# X_test = X_test_full.select_dtypes(exclude=['object'])

#split data for better validation
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, train_size=0.8,test_size=0.2)

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

X_test = imputation(X_test)
test_predict = model.predict(X_test)


output = pd.DataFrame({'PassengerId': X_test_full.PassengerId,
                       'Survived': test_predict
                      })
output.to_csv('submission_1.csv', index=False)