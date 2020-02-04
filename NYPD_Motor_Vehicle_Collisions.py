# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 23:09:06 2018

@author: AMOD-PC
"""

import os
#importing numpy array as np
import numpy as np

#importing pandas library as pd
import pandas as pd

#importing seaborn as sns
import seaborn as sns
from scipy import stats
import scipy as sc
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import statsmodels.api as sm

sns.set(style='ticks', context='talk')

pd.options.display.max_rows = 20
pd.options.display.max_columns=55


#Read CSV (comma-separated) is a function which reads data from csv file returns list of DataFrames
table = pd.read_csv("C:/Users/AMOD-PC/Desktop/Spring 18/Python/Project/NYPD_Motor_Vehicle_Collisions.csv")

table.head()

#Data Cleaning
table.drop(table.columns[7:10],axis=1,inplace=True)
table.drop(table.columns[21:],axis=1,inplace=True)

cols = table.columns
cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, (str)) else x)
table.columns = cols
table.head()



table.BOROUGH=table.BOROUGH.fillna('Unspecified')
table.ZIP_CODE=table.ZIP_CODE.fillna('')
table.CONTRIBUTING_FACTOR_VEHICLE_1=table.CONTRIBUTING_FACTOR_VEHICLE_1.fillna('Unspecified')
table.CONTRIBUTING_FACTOR_VEHICLE_2=table.CONTRIBUTING_FACTOR_VEHICLE_2.fillna('Unspecified')
table.CONTRIBUTING_FACTOR_VEHICLE_3=table.CONTRIBUTING_FACTOR_VEHICLE_3.fillna('Unspecified')
table.CONTRIBUTING_FACTOR_VEHICLE_4=table.CONTRIBUTING_FACTOR_VEHICLE_4.fillna('Unspecified')
table.CONTRIBUTING_FACTOR_VEHICLE_5=table.CONTRIBUTING_FACTOR_VEHICLE_5.fillna('Unspecified')
table.head()


table['LATITUDE']=table['LATITUDE'].fillna(0)
table['LONGITUDE']=table['LONGITUDE'].fillna(0)
table['NUMBER_OF_PERSONS_INJURED']=table['NUMBER_OF_PERSONS_INJURED'].fillna(0)
table.head()


table['Hour'] = pd.to_datetime(table['TIME']).dt.hour
table.head()


#Analysis of collisions based on borough


table_borough = table.groupby(table.BOROUGH).sum()[['NUMBER_OF_PERSONS_INJURED','NUMBER_OF_PERSONS_KILLED']]
table_borough=table_borough.drop(['Unspecified'])
table_borough

table_borough['NUMBER_OF_PERSONS_INJURED'].plot(kind='bar', title='Borough wise classification of NUMBER_OF_PERSONS_INJURED')
plt.legend(loc="upper right")

table_borough['NUMBER_OF_PERSONS_KILLED'].plot(kind='bar', 
                                               title='Borough wise classification of NUMBER_OF_PERSONS_KILLED', color="green")


#BOROUGH wise classification of number of collisions

table2=table['BOROUGH'].value_counts()
table2=table2.drop(['Unspecified'])
table2.plot.barh(title='Borough wise number of collisions reported', color="red").invert_yaxis()


#Analysis of collsions based on month of the yea
table['YEAR_DATE'] = pd.to_datetime(table['DATE'])
table['YEAR']= (table['YEAR_DATE']).dt.year
#table_year=table.groupby((table['YEAR']).dt.year).sum()[['NUMBER_OF_PERSONS_INJURED','NUMBER_OF_PERSONS_KILLED']]
#table_year


table['Months'] = pd.to_datetime(table['DATE'])
table_month=table.groupby(table.Months.dt.month).sum()[['NUMBER_OF_PERSONS_INJURED','NUMBER_OF_PERSONS_KILLED']]
table_month=table_month.reset_index()

table_month=table_month.rename({0: 'Jan', 1: 'Feb',2: 'Mar', 3: 'Apr',4: 'May', 5: 'Jun',6: 'Jul', 7: 'Aug',8: 'Sep', 9: 'Oct',10: 'Nov', 11: 'Dec'})
table_month=table_month.drop(['Months'],axis=1)

table_month



table_month['NUMBER_OF_PERSONS_INJURED'].plot.bar(title='Month wise classification of NUMBER_OF_PERSONS_INJURED')

table_month['NUMBER_OF_PERSONS_KILLED'].plot.bar(title='Month wise classification of NUMBER_OF_PERSONS_KILLED',color="green")

#Month wise classification of number of collisions
fig, ax=plt.subplots(figsize=(20,8))
ax = sns.countplot(y=table.Months.dt.month, data=table)


#Analysis of collsions based on date of the month

table['DATE'] = pd.to_datetime(table['DATE'])
table_date=table.groupby(table.DATE.dt.day).sum()[['NUMBER_OF_PERSONS_INJURED','NUMBER_OF_PERSONS_KILLED']]
table_date

table_date.plot.bar(title='Date wise classification of NUMBER_OF_PERSONS_INJURED')

#Analysis of Collisions based on hour of the day



table['HOUR'] = pd.to_datetime(table['TIME'])
table_hour=table.groupby(table.HOUR.dt.hour).sum()[['NUMBER_OF_PERSONS_INJURED','NUMBER_OF_PERSONS_KILLED']]
table_hour


table_hour['NUMBER_OF_PERSONS_INJURED'].plot.bar(title='Hour wise classification of NUMBER_OF_PERSONS_INJURED')

table_hour['NUMBER_OF_PERSONS_KILLED'].plot.bar(title='Hour wise classification of NUMBER_OF_PERSONS_KILLED',color="green")

#Hour wise classification of number of collisions
table['TIME'] = pd.to_datetime(table['TIME'])
ax = sns.countplot(x=table.TIME.dt.hour, data=table)

#Analysis based on day of the week
table['WEEKDAY'] = pd.to_datetime(table['DATE'])
table_weekday=table.groupby(table.WEEKDAY.dt.weekday_name).sum()[['NUMBER_OF_PERSONS_INJURED','NUMBER_OF_PERSONS_KILLED']]
table_weekday



ax = sns.countplot(x=table.WEEKDAY.dt.weekday_name, data=table)



#Plot hour by day of the week
plt.figure(figsize=(18, 8))
table['WEEKDAY'] = pd.to_datetime(table['DATE'])
table['HOUR'] = pd.to_datetime(table['TIME'])
dayofweek = table.groupby([table.WEEKDAY.dt.weekday_name, table.HOUR.dt.hour])['NUMBER_OF_PERSONS_INJURED'].sum().unstack().T
sns.heatmap(dayofweek)
plt.xticks(np.arange(7) + .5, ('SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'))
#plt.yticks(rotation=0)
plt.ylabel('Hour of Day\n', size=18)
plt.xlabel('\nDay of the Week', size=18)
plt.yticks(rotation=0, size=12)
plt.xticks(rotation=0, size=12)
plt.title("Number of Persons Injured Over time\n", size=18, );
         

#Analysis of Collisions based on Contributing Factor
Contributing_factor1= table['CONTRIBUTING_FACTOR_VEHICLE_1'].value_counts()
Contributing_factor2= table['CONTRIBUTING_FACTOR_VEHICLE_2'].value_counts()
Contributing_factor3= table['CONTRIBUTING_FACTOR_VEHICLE_3'].value_counts()
Contributing_factor4= table['CONTRIBUTING_FACTOR_VEHICLE_4'].value_counts()
Contributing_factor5= table['CONTRIBUTING_FACTOR_VEHICLE_5'].value_counts()

Contributing_factor = Contributing_factor1+Contributing_factor2+Contributing_factor3+Contributing_factor4+Contributing_factor5
Contributing_factor = Contributing_factor.sort_values(ascending=False).dropna()
Contributing_factor=Contributing_factor.drop(['Unspecified'])
Contributing_factor

Contributing_factor.plot(kind='barh',title='Top Contributing factors to Motor Vehicle Collisions', figsize=(10,10)).invert_yaxis()
plt.axhline(len(Contributing_factor)-18.5, color='#CC0000')



#LINEAR REGRESSION
table_reg=table.replace('Unspecified'," ")
table_reg.head()



# create a fitted model with all three features
#lm = ols(formula=' NUMBER_OF_PERSONS_INJURED ~ BOROUGH + CONTRIBUTING_FACTOR_VEHICLE_1 + Hour ', data=table_reg).fit()
lm = ols(formula=' NUMBER_OF_PERSONS_INJURED ~ BOROUGH + Hour ', data=table_reg).fit()


# print a summary of the fitted model
lm.summary()

fig, ax = plt.subplots(figsize=(12, 8))
fig = sm.graphics.plot_fit(lm, "Hour", ax=ax)

#POLYNOMIAL REGRESSION
X = np.array([0,2,4,6,8,10,12,14,16,18,20,22])
X

j=table_hour['NUMBER_OF_PERSONS_INJURED'].values

Y=np.array([5678,2553,2778,3737,8279,6448,7447,9876,10979,11443,9215,7200])



plt.grid(True)
plt.scatter(X,Y)
plt.title('')
plt.xlabel("Hours")
plt.ylabel("NUMBER_OF_PEOPLE_INJURED")


plt.show()

#with a polynomial of degree 3 , I now have a curve of best fit.
p3= np.polyfit(X,Y,3)
p1= np.polyfit(X,Y,1)
p2= np.polyfit(X,Y,2)

plt.grid(True)
plt.scatter(X,Y)
plt.xlabel("Hours")
plt.ylabel("NUMBER_OF_PEOPLE_INJURED")
plt.plot(X,np.polyval(p1,X), 'r--', label ='p1')
plt.plot(X,np.polyval(p2,X), 'm:', label='p2')

plt.plot(X,np.polyval(p3,X), 'g--', label = 'p3')
plt.legend(loc="lower right")
plt.show()

a=np.polyfit(X,Y,3)            


#Predict value at X = 12 (12 in the noon).

y_fit_12 = np.polyval(p3, 12)
print ("Y_fit Value at 12 Noon :", y_fit_12)

y_fit_9 = np.polyval(p3, 9)
print ("Y_fit Value at 9 AM :", y_fit_9)

x_fit=np.arange(24)


p3 = np.polyfit(X,Y,3)
y_fit = []
for i in range(len(x_fit)):
    y_fit.append(np.polyval(p3, i))


plt.grid(True)
plt.plot(X, Y, label = "Original")
plt.plot(x_fit, y_fit, label = "Fitted")
plt.xlabel("Hours")
plt.ylabel("NUMBER_OF_PEOPLE_INJURED")

plt.legend(loc="lower right")
plt.show()

#BOKEH PLOT
bokeh_table= table[['BOROUGH','NUMBER_OF_PERSONS_INJURED', 'LONGITUDE','LATITUDE']]

from bokeh.io import show #this command is used to import or export a file to the file system
#this command imports the tools required for axis and grids
from bokeh.models import ( GMapPlot, GMapOptions,WheelZoomTool,DataRange1d,BoxSelectTool, PanTool,
    ColumnDataSource, Circle,
    HoverTool,
    LogColorMapper
)

from bokeh.palettes import Viridis6 as palette # this command provide a collection of palettes for color mapping.
from bokeh.plotting import figure,output_file #imports the required figures like lines ,asteriks and circles for plotting data

map_options = GMapOptions(lat=40.8311959, lng =-73.93034856, map_type="roadmap", zoom=11)

plot=GMapPlot(x_range=DataRange1d(),
             y_range=DataRange1d(),
             map_options=map_options,
             api_key="AIzaSyAiQc6pK0tBpAF4f5QozuZLejDrHnOdFaA")


plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())

df = bokeh_table.filter(['NUMBER_OF_PERSONS_INJURED','LONGITUDE','LATITUDE'], axis=1)
mean = df['NUMBER_OF_PERSONS_INJURED'] >3
df = df[mean]
df


source=ColumnDataSource(data=dict(
    lat=df['LATITUDE'],
    lon=df['LONGITUDE'],
    Number_Of_Persons_Injured = df['NUMBER_OF_PERSONS_INJURED']
))


palette.reverse()

color_mapper = LogColorMapper(palette=palette) 
#providing the tools that can be used for interactive bokeh maps
TOOLS = "pan,wheel_zoom,reset,hover,save"

p = figure(
    title="New Jersey Unemployment, 2009", tools=TOOLS,
    x_axis_location=None, y_axis_location=None
)

#returns the model specified in the argument i.e Hovertool
hover = p.select_one(HoverTool)
#Whether the tooltip position should snap to the “center” (or other anchor) position of the associated glyph, or always follow the 
#current mouse cursor position.
hover.point_policy = "follow_mouse"
#hover.
tooltips = [
    ("NUMBER_OF_PERSONS_INJURED", "@Number_Of_Persons_Injured"),
    ("(Lon, Lat)", "($x, $y)"),
]




circle= Circle(x="lon",
               y="lat",
               fill_color={'field': "Number_Of_Persons_Injured",'transform': color_mapper},
               fill_alpha=0.7)
circle_renderer = plot.add_glyph(source,circle)

plot.add_tools(HoverTool(tooltips=tooltips, renderers=[circle_renderer]))



show(plot)

