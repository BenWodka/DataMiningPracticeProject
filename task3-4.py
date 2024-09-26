# Support Vector Machine (SVM)
import pandas as pd
import numpy as np
from sklearn import preprocessing
# Data Preprocessing
dfTrain = pd.read_csv('trainOri.csv',skipinitialspace=True)
dfTest = pd.read_csv('testOri.csv',skipinitialspace=True)
dfTest["income"] = dfTest["income"].str.replace(".","")
dfTrain = dfTrain[(dfTrain.values !='?').all(axis=1)]
dfTest = dfTest[(dfTest.values !='?').all(axis=1)]

# Convert to binary value for numerical attributed based on their mean value
def numericalBinary(dataset, features):
    dataset[features] = np.where(dataset[features] >= dataset[features].mean(), 1,0)

numericalBinary(dfTrain,['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week'])
numericalBinary(dfTest,['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week'])

# One-hot encoding for categorical attribute
def oneHotBind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[feature_to_encode])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop(feature_to_encode, axis=1)
    return(res)

dfTrain = oneHotBind(dfTrain,['workclass','education','marital-status','occupation','relationship','race','sex','native-country'])
dfTest  = oneHotBind(dfTest, ['workclass','education','marital-status','occupation','relationship','race','sex','native-country'])

# Add missing attributes
for attributes in dfTrain.keys():
    if attributes not in dfTest.keys():
        print("Adding missing feature {}".format(attributes))
        dfTest[attributes] = 0

dfTrain_NN = dfTrain
dfTest_NN = dfTest

def encode_income(dataset):
    le = preprocessing.LabelEncoder()
    le = le.fit(dataset['income'])
    dataset['income'] = le.transform(dataset['income'])
    return dataset

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Convert income to binary
encode_income(dfTrain)
encode_income(dfTest)

X_train = dfTrain.loc[:,dfTrain.columns !='income'].values
Y_train = dfTrain['income'].values
X_test = dfTest.loc[:,dfTest.columns !='income'].values
Y_test = dfTest['income'].values

svm = SVC(gamma='auto')
svm.fit(X_train, Y_train)
predictions = svm.predict(X_test)

print("SVM model:")
print("Accuracy: " + str(accuracy_score(Y_test, predictions)))
print(classification_report(Y_test, predictions))
######################################################################
# Neural Network
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

X_train = dfTrain_NN.loc[:,dfTrain_NN.columns !='income'].values
Y_train = dfTrain_NN['income'].values
X_test = dfTest_NN.loc[:,dfTest_NN.columns !='income'].values
Y_test = dfTest_NN['income'].values

mlp = MLPClassifier(hidden_layer_sizes=(104,104,104))
mlp.fit(X_train,Y_train)

predictions = mlp.predict(X_test)
print("=======================================================")
print("Neural Network: ")
print("Accuracy: " + str(accuracy_score(Y_test, predictions)))
print(classification_report(Y_test,predictions))