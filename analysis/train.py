# Libraries
import pandas as pd
import dill as pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

#Data Preparation
Churn_Modelling = pd.read_csv('..\data\Churn_Modelling.csv')
Churn_Modelling[["Geography","Gender"]] = Churn_Modelling[["Geography","Gender"]].astype('category')
Churn_Modelling["Geography_cat"] = Churn_Modelling["Geography"].cat.codes
Churn_Modelling["Gender_cat"] = Churn_Modelling["Gender"].cat.codes

# Split
target = Churn_Modelling["Exited"]
dataset = Churn_Modelling.iloc[:,[3,14,15,6,7,8,9,10,11,12]]
(trainX, testX, trainY, testY) = train_test_split(dataset, target, shuffle=False)

# Training model
xgb_model = XGBClassifier()
xgb_train_model = xgb_model.fit(trainX, trainY)

# Pickling
with open('../models/model_xgb.pk', 'wb') as file:
    pickle.dump(xgb_train_model, file)
with open('../models/category.pk', 'wb') as file:
    pickle.dump(category, file)  
with open('../models/column.pk', 'wb') as file:
    pickle.dump(dataset.columns.to_list(), file)