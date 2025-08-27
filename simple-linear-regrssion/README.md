# Simple Linear Regression

```
	y = mx+b
	Residual=yᵢ−y^ᵢ

```

## Importing all the libraries

```
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
```

## importing all the dataset

```
  dataset = pd.read_csv('Salary_Data.csv')
  X = dataset.iloc[:, :-1].values
  y = dataset.iloc[:, -1].values
```

## Splitting the dataset into the Training set and Test set

```
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)
```

## Training the simple linear regression model on the Training set

```
  from sklearn.linear_model import LinearRegression
  regressor = LinearRegression()
  regressor.fit(X_train, y_train)
```

## Predict the test set result

```
  y_pred = regressor.predict(X_test)
```
