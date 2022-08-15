import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
import seaborn as sns
import plotly.express as px
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

# Downloading the data
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv', sep=',')

# We remove the duplicated data
df = df_raw.copy()
df = df[df.duplicated() == False]

## Encoding the categorical variables

# Region variable convertion
def conv_region(region_name):
  if region_name == 'southwest':
    return 1
  elif region_name == 'southeast':
    return 2
  elif region_name == 'northwest':
    return 3
  elif region_name == 'northeast':
    return 4
  else:
    return 'Region sin determinar'

df['region'] = df.apply(lambda x: conv_region(x['region']), axis=1)

# Sex variable convertion
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)

# Smoker variable convertion
df['smoker'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)

# Separating the target variable (y) from the predictors(X)
X = df.drop(['charges'], axis=1)
y = df['charges']

# Spliting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
l_regr = LinearRegression()
l_regr.fit(X_train, y_train)

print('Intercept: \n', l_regr.intercept_)
print('Coefficients: \n', l_regr.coef_)
print('Score: \n', l_regr.score(X_test, y_test))

# We save the model with joblib
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../data/l_regr_model.pkl')
joblib.dump(l_regr, filename )