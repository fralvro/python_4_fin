import pandas as pd
import numpy as np
import os
import math
import xgboost as xgb
import sklearn.datasets
import sklearn.metrics
import sklearn.feature_selection
import sklearn.feature_extraction
#import sklearn.cross_validation
import sklearn.model_selection
import tqdm

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score

import seaborn as sns
import matplotlib.pyplot as plt



apps = pd.read_csv('Documents/googleplaystore.csv')

revs = pd.read_csv('Documents/googleplaystore_user_reviews.csv')

apps.dtypes

apps.info()

apps.dropna(inplace=True)

apps.info()


# DATA CLEANING 

# Change Price, Installs, Reviews, Size and Rating to float

# Price

apps.Price.unique()

apps['Price'] = apps['Price'].apply(lambda x: float(x.replace('$','')))

apps.info()

# Installs

apps['Installs'] = apps['Installs'].apply(lambda x: float(x.replace('+','').
    replace(',','')))

apps.info()

# Reviews

apps.Reviews.unique()

apps['Reviews'] = apps['Reviews'].apply(lambda x: float(x))

# Size (Now in Megas)

apps.Size.unique()

apps['Size'] = apps['Size'].apply(lambda x: x.replace('k','000'))
apps['Size'] = apps['Size'].apply(lambda x: x.replace('Varies with device','0'))
# To zero because one-hot assumes theres no type here (Beta impact is null)
apps['Size'] = apps['Size'].apply(lambda x: float(x.replace('M','')))

# Rating 

apps.Rating.unique()

apps['Rating'] = apps['Rating'].apply(lambda x: float(x))

apps.info()

# Last Updated to year 

int(apps['Last Updated'][200][-4:])

apps['Last Updated'] = apps['Last Updated'].apply(lambda x:int(x[-4:]))

apps.info()

apps['Last Updated'].unique()

#apps.to_csv('Documents/apps_clean.csv')

'''
EDA

'''

#Number of categories of apps in the store.....
def plot_number_category():
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 7)
    fig.autofmt_xdate()
    countplot=sns.categorical.countplot(apps.Category,ax=ax)
    plt.show(countplot)

plot_number_category()

# Tabular representation
top_cat=apps.groupby('Category').size().reset_index(name='Count').nlargest(10,'Count')

top_cat

top_cat.Count.sum()

# Finance is the 8th most developed Category = 3.5% of Apps

float(top_cat.loc[top_cat['Category'] == 'FINANCE']['Count'])*100/len(apps)

# What about number of Installs? 

cat=top_cat.Category.tolist()
data=apps.groupby('Category')['Installs'].agg('sum').reset_index(name='Number_Installations')

def compare_all(data):
    fig = plt.figure(figsize=(12,7))
    title=plt.title('Comparing all categories on the basis of Installs')
    bar=sns.barplot(y=data['Category'],x=data['Number_Installations'])
    plt.show(bar)
    
compare_all(data)

# Finance apps have been downloaded 0.52% of the time

float(data.loc[data['Category'] == 'FINANCE']['Number_Installations'])*100/sum(data['Number_Installations'])

display(data.nlargest(10,'Number_Installations'))


# What if we analyze each category Market. What's each type of app market size
# We create the column Revenue 

cat=top_cat.Category.tolist()
apps['Revenue'] = apps['Price']*apps['Installs']
money=apps.groupby('Category')['Revenue'].agg('sum').reset_index(name='Cat_Revenue')

def compare_rev(data):
    fig = plt.figure(figsize=(12,7))
    title=plt.title('Comparing all categories on the basis of Revenue')
    bar=sns.barplot(y=data['Category'],x=data['Cat_Revenue'])
    plt.show(bar)
    
compare_rev(money)

# Percentage of revenue of financial apps (Only by downloads) 6.6%

float(money.loc[data['Category'] == 'FINANCE']['Cat_Revenue'])*100/sum(money['Cat_Revenue'])

# Analyze Profit/Download

money['Installs'] = apps['Installs']
money['Rev_Installs'] = money['Cat_Revenue']/money['Installs']

def compare_rev_inst(data):
    fig = plt.figure(figsize=(12,7))
    title=plt.title('Comparing all categories on the basis of Revenue/Installs')
    bar=sns.barplot(y=data['Category'],x=data['Rev_Installs'])
    plt.show(bar)
    
compare_rev_inst(money)

# Correlation among variables

correl = apps[['App', 'Rating','Reviews','Size','Installs','Price','Last Updated']].corr()

correl

# Heatmap

f , ax = plt.subplots(figsize = (14,12))
title=plt.title('Correlation of Numeric Features with Installs',y=1,size=16)
heatmap=sns.heatmap(correl,square = True,  vmax=0.8)
plt.show(heatmap)

## What is it first? The Egg or the Hen?

# Analyze Apps by Content Rating

install_cont = apps.groupby('Content Rating')['Installs'].agg('sum').reset_index(name='Number_Installations')
app_content=data=apps.groupby('Content Rating')['Installs'].size().reset_index(name='Number_Apps')

def content_bar_sum(data):
    fig=plt.figure(figsize=(12,6))
    
    title=plt.title('Comparision of content ratings (Number of Installations)')
    content_bar = sns.barplot(x=data['Content Rating'],y=data['Number_Installations'])
    plt.show(content_bar)
    
def content_bar_count(data):
    fig=plt.figure(figsize=(12,6))
    
    title=plt.title('Comparision of content ratings (Number of Apps in Market)')
    content_bar = sns.barplot(x=data['Content Rating'],y=data['Number_Apps'])
    plt.show(content_bar)
    
content_bar_sum(install_cont)
content_bar_count(app_content)

## What if we create a ratio that better explains this? What about Installations / Number of Apps

content=pd.DataFrame()
content['Content Rating'] = app_content['Content Rating']
content['No_Installations/Total_Apps']=install_cont['Number_Installations']/app_content['Number_Apps']

figure=plt.figure(figsize=(12,7))
title=plt.title('Content Rating Comparision')
bar=sns.barplot(x=content['Content Rating'],y=content['No_Installations/Total_Apps'])
plt.show(bar)

## That's a different story, now we see that Teen might be a good content rating too

## What about the revenue in the different content ratings

revenue_cont = apps.groupby('Content Rating')['Revenue'].agg('sum').reset_index(name='Revenue')


fig=plt.figure(figsize=(12,6))

title=plt.title('Comparision of content ratings (Revenue)')
revenue_bar = sns.barplot(x=revenue_cont['Content Rating'],y=revenue_cont['Revenue'])
plt.show(revenue_bar)

# We can make a similar ratio: Revenue/Installs

content['Revenue/Total_Istallations']=revenue_cont['Revenue']/install_cont['Number_Installations']

figure=plt.figure(figsize=(12,7))
title=plt.title('Ratio: Revenue/Installs')
bar=sns.barplot(x=content['Content Rating'],y=content['Revenue/Total_Istallations'])
plt.show(bar)

# Another nice ratio could be Revenue/Number of Apps

content['Revenue/Total_Apps']=revenue_cont['Revenue']/app_content['Number_Apps']

figure=plt.figure(figsize=(12,7))
title=plt.title('Ratio: Revenue/Total_Apps')
bar=sns.barplot(x=content['Content Rating'],y=content['Revenue/Total_Apps'])
plt.show(bar)

# Analyze Apps by Free / Charge

install_sum_type=apps.groupby('Type')['Installs'].agg('sum').reset_index(name='Number_Installations')

def type_bar_sum(data):
    fig=plt.figure(figsize=(12,6))
    
    title=plt.title('Comparision of  types (Number of Installations)')
    content_bar = sns.barplot(x=data['Type'],y=data['Number_Installations'])
    plt.show(content_bar)
type_bar_sum(install_sum_type)

# Charge/Free ratio

100*install_sum_type.loc[1][1]/install_sum_type.loc[0][1]


# Analyze Apps by Name 

apps['Name_check']=['>2 words' if len(x.split())>2 else '<=2words' for x in apps['App'] ]

data_install= apps.groupby('Name_check')['Installs'].agg('sum').reset_index(name='Number_Installations')
data_apps= apps.groupby('Name_check').size().reset_index(name='Number_Apps')


fig,axes = plt.subplots(figsize=(15,3),ncols=2, nrows=1)

title=axes[0].set_title("No. of Installations", y = 1.1)
title=axes[1].set_title("No of Apps", y = 1.1)

plot1=sns.barplot( x=data_install['Name_check'],y=data_install['Number_Installations'] , ax=axes[0])

plot2=sns.barplot( x=data_apps['Name_check'],y=data_apps['Number_Apps'] , ax=axes[1])

plt.show(fig)

# No. of installation / No. of apps

figure=plt.figure(figsize=(12,5))
title=plt.title("Installations/Total Apps", y = 1.0)
plot3=sns.barplot( x=data_apps['Name_check'],y=data_install['Number_Installations']/data_apps['Number_Apps'] ,palette=sns.color_palette(palette="Set1",n_colors=2,desat=.8))
plt.show(figure)

'''
FOR PROJECT 2

If you were to develop an App ...

Ask yourself this questions (Besides name pick values that are present in the dataset)

    - What category of App would you choose?
    - What Name?
    - Free / Charge
    - When would you release it? (please think about a realistic year)
    - Towards whom you'll direct it (Content Rating)?
    - What will the Genre be? (Genres column)
'''

'''LINEAR REGRESSION
'''

apps = pd.read_csv('Documents/apps_clean.csv')


# Reviews as a function of Installs

x_df = pd.DataFrame(apps.Installs)
y_df = pd.DataFrame(apps.Reviews)

plt.figure(figsize=(10,8))
plt.plot(x_df, y_df,'ko')  #kx,kv,k.,ko porque son dataframes y no np arrays
plt.xlabel('Number of Installs')
plt.ylabel('Number of Reviews')

apps['log10_installs'] = apps['Installs'].apply(lambda x: math.log10(x))
apps['log10_reviews'] = apps['Reviews'].apply(lambda x: math.log10(x))

x_df = pd.DataFrame(apps.log10_installs)
y_df = pd.DataFrame(apps.log10_reviews)

plt.figure(figsize=(10,8))
plt.plot(x_df, y_df,'ko')  #kx,kv,k.,ko porque son dataframes y no np arrays
plt.xlabel('Log10 Installs')
plt.ylabel('Log10 Reviews')

# How do we find the best model from all that could exist here?

plt.figure(figsize=(10,8))
plt.plot(x_df, y_df, 'ko')
plt.plot([2, 9], [0,7], '-')
plt.plot([0, 9], [0,9], '-')
plt.plot([0, 9], [1,6], '-')
plt.xlabel('Log10 Installs')
plt.ylabel('Log10 Reviews')





# How do we find the best model from all the possible ones

# Define Gradient Descent



# Gradient Descent In our Example

## Agregamos una columna de unos porque no existe x_o para la ordenada al origen
x_df['intercept'] = 1

## Construimos una matrix para X
x = np.array(x_df)# we want this dim=2
y = np.array(y_df).flatten()  #dim1, flatten concatena arrays de arrays en uno solo continuo, chk zip
theta = np.array([0, 0]) #vector inicializado con ceros

# Función de costo 

def cost_function(X, y, theta):
    """
    Funcion de costo
    """
    ## Numero de puntos en el dataset
    m = len(y) 
    
    J = np.sum((X.dot(theta)-y)**2)/2/m # Error cuadrático medio
    
    return J


cost_function(x, y, theta)

## Gradient descent 

def gradient_descent(X, y, theta, alpha, iterations):
    
    cost_history = [0] * iterations
    
    for iteration in range(iterations):
        hypothesis = X.dot(theta)
        loss = hypothesis-y # al principio este loss es enorme porque hypothesis es un vec de ceros
        # and the first is a zeros vector because is dot of X and zeros. Este es el error que aparece en las ecuaciones de Felipe
        m = len(y) 
        gradient = X.T.dot(loss)/m  #the size of the gradient is maximized by the size of loss
        # this means that the bigger the loss (error), the more it'll move
        theta = theta - alpha*gradient
        cost = cost_function(X, y, theta) # mse
        cost_history[iteration] = cost # mse
        #print(loss)
        #print(gradient)
        
    return theta, cost_history
 
iterations = 1500 #asumimos convergencia
alpha = 0.01
    
(t, c) = gradient_descent(x,y,theta,alpha, iterations)

## Print parameters of the model beta_1,beta_0 (intercept) 
print(t)


## Ejemplo de prediccion en dos puntos x=3.5 y x=7, 
print(np.array([3.5, 1]).dot(t))
print(np.array([7, 1]).dot(t))

### What do our predictions mean? Returning to original values
### We can create a function for predicting in original magnitudes

def lin_predict(t,reviews):
    
    mod_rev = math.log10(reviews)
    pred = np.array([mod_rev, 1]).dot(t)
    pred_real = math.pow(pred,10)
    return pred_real 
    
apps.loc[123,:]
lin_predict(t,9) # A little far (Very)
np.array([3.69897, 1]).dot(t)

apps.loc[1223,:]
lin_predict(t,10000000)
np.array([7, 1]).dot(t)

math.log10(559186)
print(np.array([5.74, 1]).dot(t))
math.pow(4.02454,10)

#### Our predictions are not that good, but this is expected somehow because we are doing a l reg

## Plotting the best fit line
h_x = np.linspace(0, 9, 20) #entre 0 y 9 quiero 20 puntos regularmente espaciados
h_y = [t[1] + t[0]*xx for xx in h_x]


plt.figure(figsize=(10,6))
plt.plot(x_df.log10_installs, y_df, '.')
plt.plot(h_x, h_y, '-')
plt.axis([0,10,-2,10])
plt.xlabel('Log10 Installs')
plt.ylabel('Log10 Reviews')
plt.title('Linear Regression Model log10 Reviews Vs log10 Installs')

# Selecting alpha (Size of the step) Felipe Notes 


#        * One-Hot Encoding

# What if we have a categorical variable? (6.3 Felipe)

# Make it yes/no (0 or 1)

## Example we have Free / Paid

apps.Type.unique()

# Should we make a a column of free and a column of paid? No, unless they are not exclusive



## 

#         * Standarizing and Normalizing Data

## Do this after creating a model using othe variables and using an easier lin model

### Idea of what we did last time so that we can see the importance on having 
### similar distributions for gradient descent

## Our case suggests that we may have local minimums in where our descent could stop


sns.jointplot(apps.Installs,apps.Reviews, kind="kde");

sns.jointplot(apps.log10_installs,apps.log10_reviews, kind="kde");


## Use Felipe Notes to describe Normalization

installs_s = (apps.log10_installs - np.mean(apps.log10_installs))/np.std(apps.log10_installs)

reviews_s = (apps.log10_reviews - np.mean(apps.log10_reviews))/np.std(apps.log10_reviews)

sns.jointplot(installs_s,reviews_s, kind="kde");


#        * Polynomial Regression

#   Discussion about why it works   

## Other useful notebooks
#   imbalanced.ipynb
#   Scikit-learn and Feature Engineering
#   feature_extraction_and_selection.ipynb


'''    
    PROJECT 3
    
    - How much would you charge?
    - What is the expected rating?

'''

'''

FOR ONE HOT ENCODING

    - Types
    - Content Rating
    - Genres
    - Android Ver ? <- I would say delete it 
    - Category
    
'''

'''
FOR MACHINE LEARNING

Regression

- I'll do it to predict Price (From Name, Category, Free / Charge, Last Updated, 
Content Rating, Genre)

- You'll do it to predict Rating (From Name, Category, Free / Charge, Last Updated, 
Content Rating, Genre and **Price**)

- What goes first, rating or Installs? Or reviews?
- Remember that this case is special because we are creating a hypotetical case, not a real one.
Therefore our estimates will be weird because we are creating a kind of chain of dependent 

PROJECT 4

Let's do the regression methods we've done before but now to predict one missing value from a specific App:
    What would be the price for 'x' App if they suddently decide to charge for it

Classification 

Treating it as a missing value

- I'll do it to predict Category
- You'll do it to predict Content Rating

PROJECT 5

- From all the data you've generated estimate if you should charge for your App or not
(Exclude Price from it)

'''


'''
PROJECTS

Will continue but using different methods of regression / classification

'''





'''
FOR THE TIME SERIES PART

Interesting to group by last updated by year and get the average of the price,
then make a TS of this and predict the average price of an App if launched in 
2021

I'll make the forecast of the installs of my app for the following 3 years 
(assuming that installs reported were the number of installs at the year the apps were updated)

PROJECT 6 (or number we're at)

    - Make this forecast for your app
    - Assume you'll get a loan at 12% interest to develop it. How much money could you ask for 
    based in your predictions
    - Use the price you came up with in the previous projects to do this part.
     
    A good approach could be to see the amount of installs as a function of antiquity, tomorrow 
    will mean that it is one day old.


'''

apps['Last Updated'].describe

len(apps['Last Updated'].unique())
