import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv('petrol.csv')
# print(df)
x = df.iloc[:,0:-1].values
y = df.iloc[:,4].values
# print(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
rf = RandomForestRegressor(n_estimators=100,random_state=0)
rf2 = rf.fit(x_train,y_train)
pridict = rf.predict(x_test)
print(pridict)
print(y_test)
print(rf.score(x_test, y_test))

