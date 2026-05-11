import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = pd.read_csv("data.csv")


X = data[["temperature","humidity","pressure","wind_speed"]]


y = data["thunderstorm"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = LogisticRegression()


model.fit(X_train, y_train)


predictions = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, predictions))


new_weather = [[49, 0, 1003, 16]]

result = model.predict(new_weather)

if result[0] == 1:
    print("Thunderstorm Likely")
else:
    print("No Thunderstorm")