import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
# Data Preprocessing Q1
# 1. Remove records with unknown (?) values from both train and test data sets and remove all continuous attributes.
# Read file and remove whitespace
dfTrain = pd.read_csv('adult/adult-data.csv', header = None, names = columns, skipinitialspace=True)
dfTest = pd.read_csv('adult/adult-test.csv', header = None, names = columns, skipinitialspace=True)
#Remove "." from income
#dfTest["income"] = dfTest["income"].str.replace(".","")
# Remove question mark
dfTrain = dfTrain[(dfTrain.values !='?').all(axis=1)]
dfTest = dfTest[(dfTest.values !='?').all(axis=1)]
# Remove all continuous attributes: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
dfTrain = dfTrain.drop(['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week'], axis=1)
dfTest = dfTest.drop(['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week'], axis=1)
# Encoder method 1: use one-hot encoder
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

X_train,Y_train = dfTrain.iloc[:,1:].values, dfTrain.iloc[:, 0].values
X_test,Y_test = dfTest.iloc[:, 1:].values, dfTest.iloc[:, 0].values

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

tree = DecisionTreeClassifier()
tree.fit(X_train, Y_train)
predictions = tree.predict(X_test)

print("=======================================================")
print("Decision Tree Model:")
print("Accuracy: " + str(accuracy_score(Y_test, predictions)))
print(classification_report(Y_test, predictions))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, Y_train)
predictions = gnb.predict(X_test)

print("=======================================================")
print("Naive Bayes Model:")
print("Accuracy: " + str(accuracy_score(Y_test, predictions)))
print(classification_report(Y_test, predictions))
##################################################################
# Data Preprocessing Q2
# Read file and remove whitespace
# Remove "." from income, question mark
dfTrain = pd.read_csv('adult/adult-data.csv', header = None, names = columns, skipinitialspace=True)
dfTest = pd.read_csv('adult/adult-test.csv', header = None, names = columns, skipinitialspace=True)
dfTest["income"] = dfTest["income"].str.replace(".","")
dfTrain = dfTrain[(dfTrain.values !='?').all(axis=1)]
dfTest = dfTest[(dfTest.values !='?').all(axis=1)]
# Convert to binary value for numerical attributed based on their mean value
def numericalBinary(dataset, features):
    dataset[features] = np.where(dataset[features] >= dataset[features].mean(), 1,0)

numericalBinary(dfTrain,['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week'])
numericalBinary(dfTest,['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week'])

# one-hot encoding for categorical attribute
dfTrain = oneHotBind(dfTrain,['workclass','education','marital-status','occupation','relationship','race','sex','native-country'])
dfTest  = oneHotBind(dfTest, ['workclass','education','marital-status','occupation','relationship','race','sex','native-country'])

# Add missing attributes
for attributes in dfTrain.keys():
    if attributes not in dfTest.keys():
        print("Adding missing feature {}".format(attributes))
        dfTest[attributes] = 0

# Convert attribute income to binary
from sklearn import preprocessing
def encode_income(dataset):
    le = preprocessing.LabelEncoder()
    le = le.fit(dataset['income'])
    dataset['income'] = le.transform(dataset['income'])
    return dataset

# K-Menas
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
encode_income(dfTrain)
encode_income(dfTest)
# Specify training and testing data
X_train = dfTrain.loc[:,dfTrain.columns !='income'].values
Y_train = dfTrain['income'].values
X_test = dfTest.loc[:,dfTest.columns !='income'].values
Y_test = dfTest['income'].values
# Specify k value
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train)

print("=======================================================")
print("K-Means Model:")
score = metrics.accuracy_score(Y_train,kmeans.predict(X_train))
print('Accuracy:{0:f}'.format(score))
# Show centroid
cluster_map = pd.DataFrame()
cluster_map['data_index'] = dfTrain.index.values
cluster_map['cluster'] = kmeans.labels_
print("Centroids of the clusters:")
print(kmeans.cluster_centers_)

# KNN
from sklearn.neighbors import KNeighborsClassifier

dfTrain = pd.read_csv('adult/adult-data.csv', header = None, names = columns, skipinitialspace=True)
dfTest = pd.read_csv('adult/adult-test.csv', header = None, names = columns, skipinitialspace=True)[-11:]
dfTest["income"] = dfTest["income"].str.replace(".","")
dfTrain = dfTrain[(dfTrain.values !='?').all(axis=1)]
dfTest = dfTest[(dfTest.values !='?').all(axis=1)]

numericalBinary(dfTrain,['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week'])
numericalBinary(dfTest,['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week'])

dfTrain = oneHotBind(dfTrain,['workclass','education','marital-status','occupation','relationship','race','sex','native-country'])
dfTest  = oneHotBind(dfTest, ['workclass','education','marital-status','occupation','relationship','race','sex','native-country'])

for attributes in dfTrain.keys():
    if attributes not in dfTest.keys():
        #print("Adding missing feature {}".format(attributes))
        dfTest[attributes] = 0

# Convert income to binary
encode_income(dfTrain)
encode_income(dfTest)

X_train = dfTrain.loc[:,dfTrain.columns !='income'].values
Y_train = dfTrain['income'].values
X_test = dfTest.loc[:,dfTest.columns !='income'].values
Y_test = dfTest['income'].values

# Make predictions on validation dataset
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)

print("=======================================================")
print("KNN Model:")
print("Accuracy: " + str(accuracy_score(Y_test, predictions)))
print(classification_report(Y_test, predictions))