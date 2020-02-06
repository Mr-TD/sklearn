import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('home.csv')
# print(df)
x = df.iloc[:, 0:-1].values
y = df.iloc[:,-1].values
# print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
lr = LinearRegression()
lr.fit(x_train, y_train)
pridic = lr.predict(x_test)
print(y_test)
print(pridic)
