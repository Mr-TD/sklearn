import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv('iris.csv')
X = df.iloc[:, :-1].values
Y = df.iloc[:, 4]
label_y = LabelEncoder()
Y = label_y.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
rf = RandomForestClassifier(n_estimators=100,random_state=0)
rf2 = rf.fit(X_train,Y_train)
pridict = rf.predict(X_test)
print(pridict)
print(Y_test)
print(accuracy_score(Y_test, pridict))

