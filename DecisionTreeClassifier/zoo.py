import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

zoo = pd.read_csv('zoo.csv')
# print(zoo)
x = zoo.iloc[:,:-1].values
# print(x)
y = zoo.iloc[:,-1].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
d = DecisionTreeClassifier()
d.fit(x_train,y_train)
predict = d.predict(x_test)
print(y_test)
print(predict)
print(accuracy_score(y_test,predict))
