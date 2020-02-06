import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pd.read_csv('Digit.csv')
# print(df)
x_train = df.iloc[:210,1:].values
y_train = df.iloc[:210,0].values
x_test = df.iloc[210:,1:].values
y_test = df.iloc[210:,0].values
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
sample =x_test[300]
# print(sample)
sample.shape = (28,28)
result = dt.predict([x_test[300]])
print(result)
plt.imshow(sample,cmap='gray')
plt.show()

