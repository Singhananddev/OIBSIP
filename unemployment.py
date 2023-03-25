#!/usr/bin/env python
# coding: utf-8

# # UNEMPLOYMENT ANALYSIS WITH PYTHON
# 

# Unemployment is measured by the unemployment rate which is the number of people who are unemployed as a percentage of the total labour force. We have seen a sharp increase in the unemployment rate during Covid-19, so analyzing the unemployment rate can be a good data science project.

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# # Upload Datsets

# In[2]:


data1=pd.read_csv(r"C:\Users\ayushi\Downloads\archive (2)\Unemployment_Rate_upto_11_2020.csv")
data2=pd.read_csv(r"C:\Users\ayushi\Downloads\archive (2)\Unemployment in India.csv")


# In[3]:


data1.head()


# In[4]:


data2.head()


# # Total rows and columns in Datasets

# In[5]:


data1.shape


# In[6]:


data2.shape


# In[7]:


data1.info()


# In[8]:


data2.info()


# # Check Null Values in Datasets

# In[9]:


data1.isnull().sum()


# In[10]:


data2.isnull().sum()


# # Null Values count in Datasets

# In[11]:


data1.isnull().value_counts()


# In[12]:


data2.isnull().value_counts()


# In[13]:


data1.columns = ["States","Date","Frequency","Estimated Unemployment Rate","Estimated Employed","Estimated Labour Participation Rate","Region","lattitude","longitude"]


# In[14]:


data1


# In[15]:


data2.columns = ["States","Date","Frequency","Estimated Unemployment Rate","Estimated Employed","Estimated Labour Participation Rate","Region"]


# In[16]:


data2


# In[17]:


import plotly.express as px
df = data1 [['States','Region','Estimated Unemployment Rate']]
fig = px.sunburst(df,path=['Region','States'],values ='Estimated Unemployment Rate')
fig.show()


# # HEAT MAP

# In[18]:


plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12, 10))
sns.heatmap(data1.corr(),annot= True, cmap='crest')
plt.show()


# In[19]:


plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12, 10))
sns.heatmap(data2.corr(),annot= True,cmap='crest')
plt.show()


# In[20]:


sns.histplot(x="Estimated Employed", hue="Region", data=data1)
plt.title("Unemployment 2020")
plt.show()


# In[21]:


sns.histplot(x="Estimated Employed", hue="Region", data=data2)
plt.title("Indian Unemployment")
plt.show()


# ###### PAIR PLOT FOR Datasets

# In[22]:


sns.pairplot(data1,height=2.7, aspect=1.5,diag_kws={'color':'blue'},plot_kws={'color':'red'})
plt.show()


# In[23]:


sns.pairplot(data2,height=2.7, aspect=1.5,diag_kws={'color':'blue'},plot_kws={'color':'red'})
plt.show()


# ##### FINAL DISPLAY OF THE UNEMPLOYMENT RATE IN INDIA (red line) AND SPECIFICALLY IN 2020 IN INDIA (blue line)

# In[24]:


fig, ax = plt.subplots(figsize =(8, 7))
#labels
m=['Rural','Urban']
#title for chart
ax.set_title("\nPARTITION ACCORDING TO AREA IN \"Unemployment in india\" dataset",color="black")
# Creating color parameters
colors = ( "yellow" ,"blue")
# Creating explode data
explode = (0.2, 0.0)
# Wedge properties
wp = { 'linewidth' : 1, 'edgecolor' : "green" }
# Creating plot
wedges, texts, autotexts = ax.pie(data2['Region'].value_counts(),
                                  autopct = "%0.2f",
                                  explode = explode,
                                  labels = m,
                                  shadow = True,
                                  colors = colors,
                                  startangle = 105,
                                  wedgeprops = wp,
                                  textprops = dict(color ="brown"))
# Adding legend
ax.legend(wedges,m,
          title ="REGION",
          loc ="center left",
          bbox_to_anchor =(1, 0, 0.5, 1))
 
plt.setp(autotexts, size = 8, weight ="bold")

plt.show()


# In[25]:


data1['Region'].unique()


# In[26]:


fig, ax = plt.subplots(figsize =(8, 7))
#labels
m=['South', 'Northeast', 'East', 'West', 'North']
#title for chart
ax.set_title("\nPARTITION ACCORDING TO REGION DIRECTIONS",color="brown")
# Creating color parameters
colors = ( "orange", "cyan", "brown","grey", "indigo")
# Creating explode data
explode = (0.1, 0.0, 0.2, 0.3, 0.0)
# Wedge properties
wp = { 'linewidth' : 1, 'edgecolor' : "green" }
# Creating plot
wedges, texts, autotexts = ax.pie(data1['Region'].value_counts(),
                                  autopct = "%0.2f",
                                  explode = explode,
                                  labels = m,
                                  shadow = True,
                                  colors = colors,
                                  startangle = 90,
                                  wedgeprops = wp,
                                  textprops = dict(color ="magenta"))
# Adding legend
ax.legend(wedges,m,
          title ="Directions",
          loc ="center left",
          bbox_to_anchor =(1.2, 0, 0.5, 1))
 
plt.setp(autotexts, size = 8, weight ="bold")

plt.show()


# In[27]:


data1.columns=["state","date","frequency","estimated unemployment rate","estimated employed","estimated labour participation rate","region", "longitude", "lattitude"]


# In[28]:


print("state with most unemployment :- ",data1['state'].value_counts().idxmax())


# In[29]:


print("state with least unemployment :- ",data1['state'].value_counts().idxmin())


# In[30]:


print("Indian Region with most unemployment :- ",data1['region'].value_counts().idxmax())


# In[31]:


print("Indian Region with least unemployment :- ",data1['region'].value_counts().idxmin())


# In[32]:


import datetime as dt
import calendar

data1['date'] = pd.to_datetime(data1['date'], dayfirst=True)
data1['month_int'] =  data1['date'].dt.month
data1['month'] =  data1['month_int'].apply(lambda x: calendar.month_abbr[x])


# In[33]:


print("month with highest unemployment :- ",data1['month'].value_counts().idxmax())


# In[34]:


print("month with least unemployment :- ",data1['month'].value_counts().idxmin())


# In[35]:


sta_unemp = data1[["state","estimated unemployment rate"]].groupby("state").sum().sort_values(by="estimated unemployment rate", ascending  =False)
sta_unemp.head(10)


# In[36]:


fig=plt.figure(5)
ax0=fig.add_subplot(1,2,1)
sta_unemp[:10].plot(kind="bar",color="magenta",figsize=(15,6),ax=ax0)
ax0.set_title("Top 10 States with highest unemployment")
ax0.set_xlabel("State")
ax0.set_ylabel("Number of People Unemployed (in %)")


# In[37]:


reg_unemp = data1[["region","estimated unemployment rate"]].groupby("region").sum().sort_values(by="estimated unemployment rate", ascending  =False)
reg_unemp.head(10)


# In[38]:


#visulaization
fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
reg_unemp[:10].plot(kind="bar",color="orange",figsize=(15,6),ax=ax0)
ax0.set_title("Top Regions with highest unemployment")
ax0.set_xlabel("Region")
ax0.set_ylabel("Number of People Unemployed (in %)")


# In[39]:


# months with highest unemployment

mn_unemp = data1[["month","estimated unemployment rate"]].groupby("month").sum().sort_values(by="estimated unemployment rate", ascending  =False)
mn_unemp.head(10)


# In[40]:


#visulaization
fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
mn_unemp[:10].plot(kind="bar",color="red",figsize=(15,6),ax=ax0)
ax0.set_title("Months with highest unemployment")
ax0.set_xlabel("Region")
ax0.set_ylabel("Number of People Unemployed (in %)")


# In[41]:


# State wise rate of unemplyement

sta_emp = data1[["state","estimated employed"]].groupby("state").sum().sort_values(by="estimated employed", ascending =False)
sta_emp.head(10)
fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)

#Unemployed
sta_unemp[:10].plot(kind="bar",color="blue",figsize=(15,6),ax=ax0)
ax0.set_title("People Unemployed in each State")
ax0.set_xlabel("State")
ax0.set_ylabel("Number of People Unemployed (in %)")

#Employed
sta_emp[:10].plot(kind="bar",color="green",figsize=(15,6),ax=ax1)
ax1.set_title("People Employed in each State")
ax1.set_xlabel("State")
ax1.set_ylabel("Number of People Employed (in %)")


# In[42]:


# Region wise rate of unemplyement

reg_emp = data1[["region","estimated employed"]].groupby("region").sum().sort_values(by="estimated employed", ascending =False)
reg_emp.head(10)
fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)

#Unemployed
reg_unemp[:10].plot(kind="bar",color="blue",figsize=(15,6),ax=ax0)
ax0.set_title("People Unemployed in each Region")
ax0.set_xlabel("Region")
ax0.set_ylabel("Number of People Unemployed (in %)")

#Employed
reg_emp[:10].plot(kind="bar",color="green",figsize=(15,6),ax=ax1)
ax1.set_title("People Employed in each Region")
ax1.set_xlabel("Region")
ax1.set_ylabel("Number of People Employed (in %)")


# In[43]:


# Month wise rate of unemplyement

mn_emp = data1[["month","estimated employed"]].groupby("month").sum().sort_values(by="estimated employed", ascending =False)
mn_emp.head(10)
fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)

#Unemployed
mn_unemp[:10].plot(kind="bar",color="blue",figsize=(15,6),ax=ax0)
ax0.set_title("People Unemployed in each Month")
ax0.set_xlabel("Month")
ax0.set_ylabel("Number of People Unemployed (in %)")

#Employed
mn_emp[:10].plot(kind="bar",color="green",figsize=(15,6),ax=ax1)
ax1.set_title("People Employed in each Month")
ax1.set_xlabel("Month")
ax1.set_ylabel("Number of People Employed (in %)")


# In[44]:


# bar plot unemployment rate (monthly)

fig = px.bar(data1, x='state',y='estimated unemployment rate', animation_frame = 'month', color='state',
            title='Unemployment rate from Jan 2020 to Oct 2020 (State)')

fig.update_layout(xaxis={'categoryorder':'total descending'})

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"]=2000

fig.show()


# In[45]:


# bar plot unemployment rate (monthly)

fig = px.bar(data1, x='region',y='estimated unemployment rate', animation_frame = 'month', color='region',
            title='Unemployment rate from Jan 2020 to Oct 2020 (Region)')

fig.update_layout(xaxis={'categoryorder':'total descending'})

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"]=2000

fig.show()


# # Conclusion
# state with most unemployment :- Andhra Pradesh
# 
# state with least unemployment :- Sikkim
# 
# Indian Region with most unemployment :- North
# 
# Indian Region with least unemployment :- Northeast
# 
# month with highest unemployment :- March
# 
# month with least unemployment :- Januray
# 
# Higher The labour participation Lower the unemployment rate
# 
# 

# In[ ]:




