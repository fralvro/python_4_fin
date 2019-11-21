"""
Overall, we'd like to put together a model that could predict the number of 
wind/hail events in Kansas in the year 2020.  We could do this by looking only 
at the wind/hail events in Kansas, and using that history to build a model with
 time as the independent variable.

It might also be possible to build a model using nearby states to calculate 
events/square mile for 2020 and then multiply by the square miles in Kansas.  
It probably isn't a good idea to use coastal states in that analysis because 
their weather patterns are too different.

I would also like the students to be able to put together some graphs that 
show what is in the data.  From the NOAA data this would include things like:
    
Number of weather events in Kansas by year
Number of weather events in Kansas by month
Number of wind/hail weather events in Kansas by year
Number of wind/hail events in Tornado Alley by year
Anything else you could think of?


From the PCS data this would include:
Number of weather events in Kansas by year
Number of weather events in Kansas by month
Total losses in Kansas by year
Total losses in Tornado Alley by year
Tornado Alley should include: Texas, Oklahoma, Kansas and Nebraska.

I hope that gives you a start.  Feel free to ask questions.

Thanks so much!
"""

import pandas as pd
import numpy as np
import os
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot
import math  
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model


os.getcwd()

battle = pd.read_csv('Documents/challenge/2_boe_noaa_data.csv') 


# Some exploratory:
"""
Number of weather events in Kansas by year
Number of weather events in Kansas by month
Number of wind/hail weather events in Kansas by year
Number of wind/hail events in Tornado Alley by year
"""

battle.head()

# Change to datetime
battle['BEGIN_DATETIME'] = pd.to_datetime(battle['BEGIN_DATETIME'])
battle['END_DATETIME'] = pd.to_datetime(battle['END_DATETIME'])
battle['YEAR']= pd.to_datetime(battle['BEGIN_DATETIME'].dt.year,format='%Y')

#battle['YEAR'] = battle['BEGIN_DATETIME'].dt.year

# Number of weather events in Kansas by year

kansas = battle[battle['STATE']=='KANSAS']

kansas['YEAR']= pd.to_datetime(kansas['BEGIN_DATETIME'].dt.year,format='%Y')

yearly_count = kansas.groupby(kansas['BEGIN_DATETIME'].dt.year).count()

f, ax = plt.subplots(figsize=(11, 6))
sns.set_color_codes("pastel")
sns.barplot(x=list(yearly_count.index),y=yearly_count["EPISODE_ID"])

ax.set(ylabel="Number of Weather Events",
       xlabel="Year")
sns.despine(left=True, bottom=True)

# Number of High Wind/Hail weather events in Kansas by year 

year_wh = kansas[kansas['EVENT_TYPE'].isin(['Hail','High Wind'])].groupby(['YEAR']).count()

### General plot

f, ax = plt.subplots(figsize=(11, 6))
sns.set_color_codes("pastel")
sns.barplot(x=list(yearly_count.index),y=year_wh["EPISODE_ID"])

ax.set(ylabel="Number of Hail and Wind Events",
       xlabel="Year")
sns.despine(left=True, bottom=True)

### Differentiated

diff_event = kansas[kansas['EVENT_TYPE'].isin(['Hail','High Wind'])].groupby(['YEAR',
       'EVENT_TYPE']).count().reset_index()

f, ax = plt.subplots(figsize=(8, 5))    
sns.lineplot(x="YEAR", y="EPISODE_ID",hue='EVENT_TYPE',
                   sizes=(.25, 2.5), data=diff_event)
ax.set(ylabel="Number of Events",
       xlabel="Year")
sns.despine(left=True, bottom=True)



# Number of weather events in Kansas by month

kansas['Month'] = kansas['BEGIN_DATETIME'].dt.month

kans = kansas.groupby(['YEAR','Month']).count().reset_index()

f, ax = plt.subplots(figsize=(8, 6)) # Adjust sizes in Notebook
sns.boxplot(x=kans['Month'], y=kans['EPISODE_ID'], data=kans)
ax.set(ylabel="Number of Weather Events",
       xlabel="Month")
sns.despine(left=True, bottom=True)

# Number of High Wind/Hail weather events in Kansas by month

kans = kansas[kansas['EVENT_TYPE'].isin(['Hail','High Wind'])].groupby(['YEAR','Month']).count().reset_index()

f, ax = plt.subplots(figsize=(8, 6)) # Adjust sizes in Notebook
sns.boxplot(x=kans['Month'], y=kans['EPISODE_ID'], data=kans)
ax.set(ylabel="Number of High Wind/Hail Events",
       xlabel="Month")
sns.despine(left=True, bottom=True)


# Number of wind/hail events in Tornado Alley by year
# is not in CZ_NAME
# What do you mean by it?


# INTERESTING TO COMPARE STATES

accum = battle.groupby(['YEAR','STATE']).count().reset_index()

f, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(x="YEAR", y="EVENT_ID",
                   hue="STATE", data=accum)
ax.set(ylabel="Number of Weather Events",
       xlabel="Year")
sns.despine(left=True, bottom=True)


"""
Overall, we'd like to put together a model that could predict the number of 
wind/hail events in Kansas in the year 2020.  We could do this by looking only 
at the wind/hail events in Kansas, and using that history to build a model with
time as the independent variable.
"""


# MODEL 1
# Idea 1: Time series of wind/hail events in Kansas 
# Idea 2: Predict Number of events for 2020


# Time Series is not stationary- Look at the trend

ax = year_wh.loc[:,'EVENT_ID'].plot(marker='o', linestyle='-')
ax.set_ylabel('Number of High Wind/Hail Events');
ax.set_xlabel('Year');

# Trend (Rolling window mean)

year_wh['3YR'] = year_wh['EVENT_ID'].rolling(window=3).mean()
year_wh['5YR'] = year_wh['EVENT_ID'].rolling(window=5).mean()

year_wh[['EVENT_ID', '3YR', '5YR']].plot(figsize=(8, 5), grid=True)

### Autocorrelation

### Positive correlation with the first 4 lags but no significant correlation at all

autocorrelation_plot(year_wh['EVENT_ID'])

### Setting ARIMA MODEL

X = year_wh['EVENT_ID'].values # We have 21 values 

model = ARIMA(X, order=(2,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# Interpret this results

## PLotting the model residuals: suggesting that there may still be some trend 
#information not captured by the model.

residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()

print(residuals.describe()) # Here we can see that there is a bias in the 
# prediction (a non-zero mean in the residuals)



### ARIMA MODEL (Rolling Forecast)
### My p, d and q parameters are:...



size = int(len(X) * 0.66) # We train with 13 and test with 8
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(2,1,0)) 
	model_fit = model.fit(disp=0)
	output = model_fit.forecast(steps=2)[0]
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error_mse = mean_squared_error(test, predictions)
error_mae = mean_absolute_error(test, predictions)

"""
Why R2 not useful 

So we tend to evaluate a time-series model based more on how well it predicts 
future values, than how well it fits past values. But the R2 mainly reflects the 
latter, not the former. 
"""

r2 = r2_score(test, predictions) # Negative means that it is worst than using the average
root_mse = math.sqrt(error_mse)

# Compare error to mean 

year_wh['EVENT_ID'].values.mean()


# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

# Predict 


# create a differenced series

# fit model
model = ARIMA(X, order=(2,1,0))
model_fit = model.fit(disp=0)
# multi-step out-of-sample forecast
forecast = model_fit.forecast(steps=2)[0]

back = np.full(21, np.nan)
forecast = np.insert(forecast, 0,X[len(X)-1])
fore = np.concatenate((np.full(20, np.nan),forecast), axis=0)

pyplot.plot(X)
pyplot.plot(fore, color='red')
pyplot.show()

# Predicted value for 2020

forecast[2]



# MODEL 2
# Idea 1: Divide number of events over size of state and then multiply by size



# Model 3
# Idea 1: Do the same as time series but including time as one of the variables
# Idea 2: Try XGBoost

# Linear and Polinomial Regression


year_wh = year_wh.reset_index()

year_wh['YEAR']= year_wh['YEAR'].dt.year
year_wh['EVENTS']= year_wh['EVENT_ID']


## We know from plot that we don't have a linear relationship between time and events


## EXPLAIN Train and Test Set: Why we need this? To have a better perspective of how 
# our model performs in real life. Not testing with what it trained

y = pd.DataFrame(year_wh['EVENTS'])
x = pd.DataFrame(year_wh['YEAR'])

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33,random_state=42)

regr = linear_model.LinearRegression()

regr.fit(x_train,y_train)

y_pred = regr.predict(x_test)

regr.coef_

mse_lin = mean_squared_error(y_test, y_pred)
mse_lin

root_mse_lin = math.sqrt(mse_lin)
root_mse_lin

mae_lin = mean_absolute_error(y_test, y_pred)

### Plots of Training

plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)

plt.xlabel(xlabel='Year')
plt.ylabel(ylabel='Events')
plt.title(label='Linear Regression in Test Set')
plt.show()


### Plot of model 

y_plot = regr.predict(x)

plt.scatter(x, y,  color='black')
plt.plot(x, y_plot, color='blue', linewidth=3)

plt.xlabel(xlabel='Year')
plt.ylabel(ylabel='Events')
plt.title(label='Linear Regression in all Data')
plt.show()


## Predict 2020 

goal = np.array([[2020]])
pred_2020 = regr.predict(goal)

float(pred_2020)


# Polynomial 

y = year_wh['EVENTS']
x = year_wh['YEAR']

degree = 5
z = np.polyfit(x, y, degree)

# See coeffs
z

ypred = np.polyval(z,x)



plt.plot( x, y,marker='o', markerfacecolor='blue', markersize=3, color='skyblue', linewidth=2)
plt.plot( x, ypred, marker='', color='olive', linewidth=2)
plt.xlabel(xlabel='Year')
plt.ylabel(ylabel='Events')
plt.title(label='Polynomial Regression')
plt.show()


r2 = r2_score(y.values, ypred) 

mse_pol = mean_squared_error(y.values, ypred)
mse_pol

root_mse_pol = math.sqrt(mse_lin)
root_mse_pol

mae_pol = mean_absolute_error(y.values, ypred)

# Get Optimal Polynomial Regression

# Remember Train and Test (Making our model consistent always, not only this case)


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33,random_state=42)

errors = []
for i in range(50):
    degree = i
    z = np.polyfit(x_train, y_train, degree)
    
    # See coeffs
    z
    
    ypred = np.polyval(z,x_test)
    
    mae_pol_it = mean_absolute_error(y_test.values, ypred)
    
    errors.append(mae_pol_it)


degree = 3
z = np.polyfit(x_train, y_train, degree)

# See coeffs
z

ypred = np.polyval(z,x_test)

mae_pol_it = mean_absolute_error(y_test.values, ypred)

errors.append(mae_pol_it)


# Predict the year we want (2020)

pred_2020 = np.polyval(z,2020)

pred_2020