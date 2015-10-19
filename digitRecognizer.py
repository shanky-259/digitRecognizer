
#  Feature selection using random forests, selected features passed to logistic regression model

import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


#building dataframes from csv
trainDF = pd.read_csv('train.csv', header = 0)
testDF = pd.read_csv('test.csv', header = 0)

#building numpy array from the dataframes
trainNumpy = trainDF.values
testNumpy = testDF.values

#model building and selecting important features
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(trainNumpy[0::, 1::], trainNumpy[0::, 0])
allFeatureIndices = forest.feature_importances_
selectedFeatureIndices = allFeatureIndices.argsort()[::-1][:400]
#print selectedFeatureIndices

#building new dataframes based on important features
trainColumns = trainDF.columns.values
modTrainDF = pd.DataFrame(trainDF['label'])
for i in range(len(trainColumns)):
    if i in selectedFeatureIndices:
        modTrainDF[trainColumns[i]] = trainDF[trainColumns[i]]

testColumns = testDF.columns.values
modTestDF = pd.DataFrame()
for i in range(len(testColumns)):
    if i in selectedFeatureIndices:
        modTestDF[testColumns[i]] = testDF[testColumns[i]]

#building numpy array from the new dataframes
modTrainNumpy = modTrainDF.values
modTestNumpy = modTestDF.values
#print(modTrainDF.columns.values)
print("start logistic regression model building..")

#finally, building the model and generating output

logreg = LogisticRegression(penalty = "l1", C = 1.0)
logreg = logreg.fit(modTrainNumpy[0::, 1::], modTrainNumpy[0::, 0])
output = logreg.predict(modTestNumpy)

#write to a csv
ImageId = [(i+1) for i in range(len(output))]
outputDF = pd.DataFrame(ImageId, columns = ['ImageId'])
outputDF['Label'] = output
#print output_DF.head()
outputDF.to_csv('predictionfileModified1.csv', index = False)
