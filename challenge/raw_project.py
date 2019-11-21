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

#battle['YEAR'] = battle['BEGIN_DATETIME'].dt.year

# Number of weather events in Kansas by year

kansas = battle[battle['STATE']=='KANSAS']

kansas['YEAR']= kansas['BEGIN_DATETIME'].dt.year

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

f, ax = plt.subplots(figsize=(5, 5))    
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

# Number of wind/hail events in Tornado Alley by year
# is not in CZ_NAME
# What do you mean by it?

#kansas[kansas['EVENT_TYPE'].isin(['Hail','High Wind']),kansas['EVENT_TYPE']==']

# INTERESTING TO COMPARE STATES



palette = sns.color_palette("mako_r", 6)
sns.lineplot(x=list(yearly_count.index), y="EPISODE_ID", palette=palette, data=yearly_count)

plt.bar(yearly_count.index, yearly_count['EPISODE_ID'])
# Rotation of the bars names
plt.xticks(yearly_count.index, bars, rotation=90)

"""
Overall, we'd like to put together a model that could predict the number of 
wind/hail events in Kansas in the year 2020.  We could do this by looking only 
at the wind/hail events in Kansas, and using that history to build a model with
time as the independent variable.
"""


# MODEL 1
# Idea 1: Time series of wind/hail events in Kansas 
# Idea 2: Predict Number of events for 2020


# MODEL 2
# Idea 1: Divide number of events over size of state and then multiply by size

# Model 3
# Idea 1: Do the same as time series but including time as one of the variables
# Idea 2: Try XGBoost







