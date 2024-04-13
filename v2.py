from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

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

# X_full, X_test_full = drop_columns_missing_values(X_full, X_test_full)

# Ordinal Encoding
# X , X_test = ordinal_encoding(X_full, X_test_full)

#split data for better validation
train_X, val_X, train_y, val_y = train_test_split(X_full, y, random_state=1, train_size=0.8,test_size=0.2)


#select categorical data 
categorical_cols = [cname for cname in X_full.columns if X_full[cname].nunique() < 10 and 
                        X_full[cname].dtype == "object"]

#select numerical data
numerical_cols = [cname for cname in X_full.columns if X_full[cname].dtype in ['int64', 'float64']]

cols = categorical_cols + numerical_cols
train_X = train_X[cols].copy()
val_X = val_X[cols].copy()
X_test = X_test_full[cols].copy()
X = X_full[cols].copy()

#one hot
train_X = pd.get_dummies(train_X)
val_X = pd.get_dummies(val_X)
X_test = pd.get_dummies(X_test)
X = pd.get_dummies(X)

train_X, val_X = train_X.align(val_X, join='left', axis=1)
train_X, X_test = train_X.align(X_test, join='left', axis=1)
X, X_test = X.align(X_test, join='left', axis=1)

# #deal with missing values
# train_X = imputation(train_X)
# val_X = imputation(val_X)

#model
model = XGBRegressor(n_estimators = 400, random_state=1, early_stopping_rounds = 5,
                     learning_rate = 0.04, n_jobs=5)

model.fit(X, y, eval_set = [(val_X,val_y),(train_X,train_y)], verbose = False)
test_predict = model.predict(X_test)
test_predict = [round(i) for i in test_predict]

output = pd.DataFrame({'PassengerId': X_test_full.PassengerId,
                       'Survived': test_predict
                      })
output.to_csv('submission_XgbR.csv', index=False)