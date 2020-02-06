import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('salary.csv')
# print(df)
l1 = LabelEncoder()
new1 = l1.fit_transform(df['company'])
df['company'] = new1
l2 = LabelEncoder()
new2 = l2.fit_transform(df['job'])
df['job'] = new2
l3 = LabelEncoder()
new3 = l3.fit_transform(df['degree'])
df['degree'] = new3
x = df.iloc[:,0:-1].values
y = df.iloc[:,-1].values
# print(df)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
r = RandomForestClassifier(n_estimators=100,random_state=0)
result = r.fit(x_train,y_train)
predict = r.predict(x_test)
print(y_test)
print(predict)
