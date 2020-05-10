# Libraries
import pandas as pd
import dill as pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Data Preparation
Churn_Modelling = pd.read_csv('..\data\Churn_Modelling.csv')
Churn_Modelling[["Geography","Gender"]] = Churn_Modelling[["Geography","Gender"]].astype('category')
Churn_Modelling["Geography_cat"] = Churn_Modelling["Geography"].cat.codes
Churn_Modelling["Gender_cat"] = Churn_Modelling["Gender"].cat.codes

# Split
target = Churn_Modelling["Exited"]
dataset = Churn_Modelling.iloc[:,[14,15,6,7,8,9,10,11,12]]
(trainX, testX, trainY, testY) = train_test_split(dataset, target, shuffle=False)

# Training model
model = RandomForestClassifier(n_estimators=100)
model.fit(trainX, trainY)
predictions = model.predict(testX)

# Pickling
filename = 'model_rf.pk'
with open('../models/'+filename, 'wb') as file:
    pickle.dump(model, file)