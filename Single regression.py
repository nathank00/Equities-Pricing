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

end1 = datetime.date(2022, 8, 5)
start1 = end1 - pd.Timedelta(days = 365 * 5)

AAPL_df = yf.download("AAPL", start = start1, end = end1, progress = False)
MSFT_df = yf.download("MSFT", start = start1, end = end1, progress = False)
#pep_df = yf.download("PEP", start = start1, end = end1, progress = False)
#usdx_df = yf.download("DX-Y.NYB", start = start1, end = end1, progress = False)

## Calculate Log Returns based on Adjusted Close Prices

AAPL_df['AAPL'] = np.log(AAPL_df['Adj Close'] / AAPL_df['Adj Close'].shift(1))
MSFT_df['MSFT'] = np.log(MSFT_df['Adj Close'] / MSFT_df['Adj Close'].shift(1))
#pep_df['pep'] = np.log(pep_df['Adj Close'] / pep_df['Adj Close'].shift(1))
#usdx_df['usdx'] = np.log(usdx_df['Adj Close'] / usdx_df['Adj Close'].shift(1))

## Create Dataframe with X variables and Y variable

df = pd.concat([MSFT_df['MSFT'], AAPL_df['AAPL']], axis = 1).dropna()

#Saving the Dataframe as a csv file

df.to_csv("Jul2022_data_lin_regression.csv")

## Scatter Plot of MSFT Returns vs SH returns

plt.figure(figsize = (10, 6))
plt.rcParams.update({'font.size': 14})
plt.xlabel("MSFT returns")
plt.ylabel("AAPL returns")
plt.title("Scatter plot of daily returns (Jul 2019 to Jul 2022)")
plt.scatter(df['MSFT'], df['AAPL'])
#plt.show()

## Find Correlation between X and Y

df.corr()
print(df.corr())

### Create an instance of the class OLS
slr_sm_model = smf.ols('AAPL ~ MSFT', data = df)

### Fit the model (statsmodels calculates beta_0 and beta_1 here)
slr_sm_model_AAPL = slr_sm_model.fit()

### Summarize the model

print(slr_sm_model_AAPL.summary())

param_slr = slr_sm_model_AAPL.params

## Print the parameter estimates of the simple linear regression model

print("\n")
print("====================================================================")
print("The intercept in the statsmodels regression model is", \
      np.round(param_slr.Intercept, 4))
print("The slope in the statsmodels regression model is", \
      np.round(param_slr.MSFT, 4))
print("Simple Linear Regression in equation form:", "Y_i =", np.round(param_slr.Intercept, 4), "+", np.round(param_slr.MSFT, 4), "X_i")
print("====================================================================")
print("\n")

## Linear regression plot of X (MSFT) and Y (SH)

plt.figure(figsize = (10, 6))
plt.rcParams.update({'font.size': 14})
plt.xlabel("MSFT returns")
plt.ylabel("AAPL returns")
plt.title("Simple linear regression model")
#plt.scatter(df['MSFT'],df['AAPL'])
plt.plot(df['MSFT'], param_slr.Intercept+param_slr.MSFT * df['MSFT'],
         label='Y={:.4f}+{:.4f}X'.format(param_slr.Intercept, param_slr.MSFT),
         color='red')
plt.legend()

## Create an instance of the class LinearRegression()

slr_skl_model = LinearRegression()

## Fit the model (sklearn calculates beta_0 and beta_1 here)

X = df['MSFT'].values.reshape(-1, 1)
slr_skl_model_AAPL = slr_skl_model.fit(X, df['AAPL'])

print("The intercept in the sklearn regression result is", \
      np.round(slr_skl_model_AAPL.intercept_, 4))
print("The slope in the sklearn regression model is", \
      np.round(slr_skl_model_AAPL.coef_[0], 4))


## Linear regression plot of X (MSFT) and Y (ko)

plt.figure(figsize = (10, 6))
plt.rcParams.update({'font.size': 14})
plt.xlabel("MSFT returns")
plt.ylabel("AAPL returns")
plt.title("Simple linear regression model")
plt.scatter(df['MSFT'], df['AAPL'])
plt.plot(X, slr_skl_model.predict(X),
         label='Y={:.4f}+{:.4f}X'.format(slr_skl_model.intercept_, \
                                         slr_skl_model.coef_[0]),
         color='red')
plt.legend()
plt.show()