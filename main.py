import pandas as pd
import statsmodels.regression.linear_model as lm
import statsmodels.tools.tools as ct
import statsmodels.formula.api as smf
import yfinance as yf
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## Obtain historical data

end1 = datetime.date(2022, 7, 28)
start1 = end1 - pd.Timedelta(days = 365 * 3)

ko_df = yf.download("KO", start = start1, end = end1, progress = False)
spy_df = yf.download("SPY", start = start1, end = end1, progress = False)
pep_df = yf.download("PEP", start = start1, end = end1, progress = False)
usdx_df = yf.download("DX-Y.NYB", start = start1, end = end1, progress = False)

## Calculate Log Returns based on Adjusted Close Prices

ko_df['ko'] = np.log(ko_df['Adj Close'] / ko_df['Adj Close'].shift(1))
spy_df['spy'] = np.log(spy_df['Adj Close'] / spy_df['Adj Close'].shift(1))
pep_df['pep'] = np.log(pep_df['Adj Close'] / pep_df['Adj Close'].shift(1))
usdx_df['usdx'] = np.log(usdx_df['Adj Close'] / usdx_df['Adj Close'].shift(1))

## Create Dataframe with X variables and Y variables

df = pd.concat([spy_df['spy'], ko_df['ko'], pep_df['pep'], usdx_df['usdx']], axis = 1).dropna()

#Saving the Dataframe as a csv file

df.to_csv("Jul2022_data_lin_regression.csv")

## Scatter Plot of SPY Returns vs ko returns

plt.figure(figsize = (10, 6))
plt.rcParams.update({'font.size': 14})
plt.xlabel("SPY returns")
plt.ylabel("ko returns")
plt.title("Scatter plot of daily returns (Jul 2019 to Jul 2022)")
plt.scatter(df['spy'], df['ko'])
#plt.show()
'''
BigNate
'''
## Find Correlation between X and Y

df.corr()
#print(df.corr())

### Create an instance of the class OLS
slr_sm_model = smf.ols('ko ~ spy', data = df)

### Fit the model (statsmodels calculates beta_0 and beta_1 here)
slr_sm_model_ko = slr_sm_model.fit()

### Summarize the model

#print(slr_sm_model_ko.summary())

param_slr = slr_sm_model_ko.params

## Print the parameter estimates of the simple linear regression model

#print("\n")
#print("====================================================================")
#print("The intercept in the statsmodels regression model is", \
      #np.round(param_slr.Intercept, 4))
#print("The slope in the statsmodels regression model is", \
      #np.round(param_slr.spy, 4))
#print("Simple Linear Regression in equation form:", "Y_i =", np.round(param_slr.Intercept, 4), "+", np.round(param_slr.spy, 4), "X_i")
#print("====================================================================")
#print("\n")

## Simple Linear regression plot of X (spy) and Y (spx)

plt.figure(figsize = (10, 6))
plt.rcParams.update({'font.size': 14})
plt.xlabel("SPY returns")
plt.ylabel("ko returns")
plt.title("Simple linear regression model")
plt.scatter(df['spy'],df['ko'])
plt.plot(df['spy'], param_slr.Intercept+param_slr.spy * df['spy'],
         label='Y={:.4f}+{:.4f}X'.format(param_slr.Intercept, param_slr.spy),
         color='red')
plt.legend()





### Fit a multiple linear regression model to the data using statsmodels



### Create an instance of the class OLS
mlr_sm_model = smf.ols('ko ~ spy + pep + usdx', data = df)

### Fit the model (statsmodels calculates beta_0, beta_1, beta_2, beta_3 here)
mlr_sm_model_ko = mlr_sm_model.fit()

### Summarize the model

print(mlr_sm_model_ko.summary())

#### Print the parameter estimates of the muliple linear regression model

param_mlr = mlr_sm_model_ko.params

print("\n")
print("====================================================================")
print("The intercept and slopes in the statsmodels regression model are")
print("\n")
print(param_mlr)
print("====================================================================")
print("\n")

## Create an instance of the class LinearRegression()

slr_skl_model = LinearRegression()

## Fit the model (sklearn calculates beta_0 and beta_1 here)

X = df['spy'].values.reshape(-1, 1)
slr_skl_model_ko = slr_skl_model.fit(X, df['ko'])

print("The intercept in the sklearn regression result is", \
      np.round(slr_skl_model_ko.intercept_, 4))
print("The slope in the sklearn regression model is", \
      np.round(slr_skl_model_ko.coef_[0], 4))

## Linear regression plot of X (spy) and Y (ko)

plt.figure(figsize = (10, 6))
plt.rcParams.update({'font.size': 14})
plt.xlabel("SPY returns")
plt.ylabel("KO returns")
plt.title("Simple linear regression model")
plt.scatter(df['spy'], df['ko'])
plt.plot(X, slr_skl_model.predict(X),
         label='Y={:.4f}+{:.4f}X'.format(slr_skl_model_ko.intercept_, \
                                         slr_skl_model_ko.coef_[0]),
         color='red')
plt.legend()
plt.show()
