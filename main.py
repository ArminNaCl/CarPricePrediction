import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("CarPrice.csv")

predict = "price"

data = data[
    [
        "symboling",
        "wheelbase",
        "carlength",
        "carwidth",
        "carheight",
        "curbweight",
        "enginesize",
        "boreratio",
        "stroke",
        "compressionratio",
        "horsepower",
        "peakrpm",
        "citympg",
        "highwaympg",
        "price",
    ]
]

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)


model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)
model.score(xtest, predictions)
