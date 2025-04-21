# EX:4 SGD-Regressor-for-Multivariate-Linear-Regression
# NAME: INFANTINA MARIA L
# REG NO: 212223100013
## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries-Load necessary Python libraries.
2. Load Data-Read the dataset containing house details.
3. Preprocess Data-Clean and solit the data into training and testing sets.
4. Select Features & Target-Choose input variables(features) and output variables(house price,occupants).
5. Train Mode-Use SGDRegressor() to train the model.
6. Make Predictions-Use the model to predict house price and occupants.
7. Evaluate Performance-Check accuracy using error metrics.
8. Improve Model-Tune settings for better accuracy.

## Program:

Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
# Developed by: INFANTINA MARIA L
# RegisterNumber: 212223100013

```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data=fetch_california_housing()
print(data)
```
![image](https://github.com/user-attachments/assets/4c315cbf-af66-490d-a5a3-765dc6e4d32f)
```
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
print(df.tail())
print(df.info())
```
![image](https://github.com/user-attachments/assets/2022acfa-433a-4827-91f1-74ccff87a427)
```
x=df.drop(columns=['AveOccup','target'])
x.info()
x.shape
```
![image](https://github.com/user-attachments/assets/38b17a6e-302b-44f1-bba5-4776b75b8f37)
```
y=df[['AveOccup','target']]
y.info()
y.shape
```
![image](https://github.com/user-attachments/assets/8d590002-356b-4775-aa9f-fec7f507a0ff)
```
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x.head()
```
![image](https://github.com/user-attachments/assets/83b2fa73-664c-4848-bfba-2c771c3e7420)
```
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
```
![image](https://github.com/user-attachments/assets/fda38679-5f7d-42bc-aa03-86c414d08faf)
```
scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.transform(x_test)
y_train=scaler_y.fit_transform(y_train)
y_test=scaler_y.transform(y_test)
print(x_train)

```
![image](https://github.com/user-attachments/assets/ec2f4922-c6ce-418e-a456-0e8e7ccc444d)
```
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_sgd=MultiOutputRegressor(sgd)
multi_sgd.fit(x_train,y_train)

```
![image](https://github.com/user-attachments/assets/5a3563ec-c887-44a5-92a8-6327b346e9b1)
```
y_pred=multi_sgd.predict(x_test)
y_pred=scaler_y.inverse_transform(y_pred)
y_test=scaler_y.inverse_transform(y_test)
mse=mean_squared_error(y_test,y_pred)
print("Mean Squareed Error:",mse)
```
![image](https://github.com/user-attachments/assets/58efb14c-7662-46c5-b5d3-70147b93c445)
```
print(y_pred[:5])
```
![image](https://github.com/user-attachments/assets/a0d5ab0a-58eb-4875-b9ef-1b5c640834ad)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
