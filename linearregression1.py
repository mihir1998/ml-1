import pandas as pd
import pickle 


df = pd.read_csv("N:/flask-ml-1/linear-regression.csv")
df2 = df.fillna(value = df["age"].mean())
df2

x = df2.drop("price", axis="columns")



y = df2.price
y

from sklearn.linear_model import LinearRegression

lin = LinearRegression()
lin.fit(x, y)

pre = lin.predict([[3000, 2, 18]])
print(pre)

pickle.dump(lin, open("model.pkl", "wb"))

model = pickle.load(open("model.pkl", "rb"))
m = model.predict([[3000, 2, 18]])
print(m)

area = 3000
room = 5
age = 10
m2 = model.predict([[area, room, age]])