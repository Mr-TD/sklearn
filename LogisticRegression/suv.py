import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv('suv.csv')
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:, 4]
label_y = LabelEncoder()
Y = label_y.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
'''
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)
'''
logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)
pridic = logmodel.predict(X_test)
print(Y_test)
print(pridic)
#print(dataset)
result = confusion_matrix(Y_test,pridic)
print(result)
print((80/22)*10)
