# machine-learning process

1. Data pre-processing

- Import data
- Clean the data
- Split into training & test set

2. Modeling

- Build the model
- Train the model
- Prediction

3. Evaluation

- Calculate performance metrics
- Make a verdict

**Feature Scaling is always applied to columns** <br>

## Types of feature scaling <br>

| `Normalization` | X' = (X-Xmin) / (Xmax-Xmin) | `[0 or 1]` | <br>
| `Standardization` | X' = (X - η) / σ | `[-3; +3]`

# Data Preprocessing Tools

**Importing the Libraries**

```
	import numpy as np
 	import matplotlib.pyplot as plt
 	import pandas as pd
```

** Importing the dataset**

```
	dataset = pd.read_csv('Data.csv')
	x = dataset.iloc[:, :-1].values
	y = dataset.iloc[:, -1].values
```

# for print

```
	print(x)
	print(y)
```

# Taking care of missing Data

**All the missing column would change into average data from that numeric column. This code only works with numerical column**

```
	from sklearn.impute import SimpleImputer
	imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
	imputer.fit(X[:, 1:3])
	X[:,1:3] = imputer.transform(X[:,1:3])
	print(X)
```

# Encoding categorical data

### Encoding the independent variable

## Using onehotencoder for transform country name into numbers

```
	from sklearn.compose import ColumnTransformer
	from sklearn.preprocessing import OneHotEncoder
	ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
	X = np.array(ct.fit_transform(X))

```

### Encoding the dependent variable

```
	from sklearn.preprocessing import LabelEncoder
	le = LabelEncoder()
	Y = le.fit_transform(Y) //making purchase column into binary
```
