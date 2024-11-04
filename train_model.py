import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import joblib 
import os 
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
data=pd.read_csv("medical_insurance.csv")
data.columns
data.shape
data.dtypes
data.isnull().sum()
duplicates=data.duplicated().sum()
data.shape
print("Number of duplicate rows:", duplicates)
data=data.drop_duplicates()
data.duplicated().sum()
numerical_columns = ['age', 'bmi', 'children', 'charges']

plt.figure(figsize=(12, 8))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data[column])
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply the function to each numerical column to remove outliers
for col in numerical_columns:
    data = remove_outliers(data, col)
    
print(data.info())
print(data.describe())
data.shape
numerical_columns = ['age', 'bmi', 'children', 'charges']

plt.figure(figsize=(12, 8))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data[column])
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()
varquanti=['age','bmi', 'children', 'charges']
for col in varquanti:
    data[col].hist(bins=20,figsize=(3,3))
    plt.show()
    
y = data["charges"]
x = data.drop(["charges"], axis=1)
x_encoded = pd.get_dummies(x, columns=['sex', 'smoker', 'region'], drop_first=True)

binary_data=x_encoded[["smoker_yes","sex_male","region_northwest","region_southeast","region_southwest"]]
non_binary_data=x_encoded.drop(columns=["smoker_yes", "sex_male", "region_northwest", "region_southeast", "region_southwest"])

Scaler=StandardScaler()
xs=Scaler.fit_transform(non_binary_data) 
xs_scaled=pd.DataFrame(xs,columns=non_binary_data.columns,index=non_binary_data.index)
xs_scaled.head()

joblib.dump(Scaler,"scaler.pkl")
Xs_final = pd.concat([xs_scaled, binary_data], axis=1)
print(Xs_final.head())

x_train, x_test,y_train,y_test=train_test_split(Xs_final,y,test_size=0.2,random_state=123)

print(x_train.shape)
print(x_test.shape)

model=LinearRegression()
model.fit(x_train,y_train)
model.coef_
model.intercept_

pred=model.predict(x_test)
MSE=mean_squared_error(pred,y_test)
print(MSE)

plt.scatter(y_test,pred)
plt.xlabel("True_values")
plt.ylabel("predicted values")

model=Ridge()
# Préciser les valeurs de lambda à tester 
nb_va=300
# donne des valeurs entre 10^(-3) et 10^(2)
lambda_values=np.logspace(-3,2,nb_va)
lambda_range={"alpha":lambda_values}
model=Ridge()

grid=GridSearchCV(model,lambda_range,scoring="neg_mean_squared_error",
                  cv=5)

grid.fit(x_train,y_train)
grid.best_params_
model1=Ridge(alpha=0.57)
model1.fit(x_train,y_train)
print(x_train.columns)
print(model1.coef_)
pred2=model1.predict(x_test)
MSE=mean_squared_error(pred2,y_test)
R2_score=r2_score(pred2,y_test)
print(MSE.round(2))
print(R2_score)

plt.scatter(y_test, pred2)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs Predicted Values")
plt.show()

nb_va=300
lambda_values=np.logspace(-3,2,nb_va)
lambda_range={"alpha":lambda_values}
model=Lasso()
grid=GridSearchCV(model,lambda_range,scoring='neg_mean_squared_error',cv=5)
grid.fit(x_train,y_train)

model2=Lasso(alpha=0.001)
model2.fit(x_train,y_train)
print(x_train.columns)
print(model2.coef_)

pred3=model2.predict(x_test)
MSE=mean_squared_error(pred3,y_test)
R2_score=r2_score(pred3,y_test)
print(MSE.round(2))
print(R2_score)

plt.scatter(y_test, pred3)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs Predicted Values")
plt.show()

nb_val=300 # nbr de valeurs à tester 
lambda_range=np.logspace(-3,2,nb_val) # valeurs de lambda à tester
rho_range=[0.1,0.01,0.001]    # valeurs de rho à tester  
hpers={"alpha":lambda_range,"l1_ratio":rho_range} 

model=ElasticNet()
grid=GridSearchCV(model,hpers,cv=5,scoring="neg_mean_squared_error")
grid.fit(x_train,y_train)

model3=ElasticNet(alpha=0.001,l1_ratio=0.001)
model3.fit(x_train,y_train)
pred=model3.predict(x_test)
MSE=mean_squared_error(pred,y_test)
R2_score=r2_score(pred,y_test)
MSE.round(2)
print(MSE)
R2_score

plt.scatter(y_test, pred)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs Predicted Values")
plt.show()


joblib.dump(model, 'ELN_model.pkl')
scaler = joblib.load('scaler.pkl')
scaler.feature_names_in_