#!/usr/bin/env python
# coding: utf-8

# # TASK 3- CAR PRICE PREDICTION WITH MACHINE LEARNING

# In[1]:


#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[2]:


df = pd.read_csv(r"C:\Users\ayushi\Downloads\CarPrice.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.describe()


# # data visualization

# In[8]:


df.columns
print(df['fueltype'].value_counts())
print(df['aspiration'].value_counts())
print(df['enginelocation'].value_counts())
fueltype = df['fueltype']
aspiration = df['aspiration']
enginelocation = df['enginelocation']
price = df['price']


# In[9]:


style.use('ggplot')
fig = plt.figure(figsize=(15,5))
fig.suptitle('Visualizing categorical data columns')
plt.subplot(1,3,1)
plt.bar(fueltype,price, color='royalblue')
plt.xlabel("fueltype")
plt.ylabel("price")
plt.subplot(1,3,2)
plt.bar(aspiration, price, color='brown')
plt.xlabel("aspiration")
plt.subplot(1,3,3)
plt.bar(enginelocation, price, color='purple')
plt.xlabel('enginelocation')
plt.show()


# In[10]:


fig, axes = plt.subplots(1,3,figsize=(15,5), sharey=True)
fig.suptitle('Visualizing categorical columns')
sns.barplot(x=fueltype, y=price, ax=axes[0])
sns.barplot(x=aspiration, y=price, ax=axes[1])
sns.barplot(x=enginelocation, y=price, ax=axes[2])
     


# In[11]:


gas_data = df.groupby('fueltype').get_group('gas')
gas_data.describe()


# In[12]:


aspiration_data = df.groupby('aspiration').get_group('std')
aspiration_data.describe()


# In[13]:


fig=plt.figure(figsize=(10 , 10))
plt.subplot(2 , 2 ,1)
sns.barplot(x='fueltype', y='price' , data=df)
plt.subplot(2 , 2 ,2)
sns.barplot(x='fueltype' , y='horsepower' , data=df)
plt.tight_layout()
plt.subplot(2 , 2 ,3)
sns.barplot(x='fueltype' , y='carlength' , data=df)
plt.subplot(2 , 2 ,4)
sns.barplot(x='fueltype' , y='carwidth' , data=df)


# In[14]:


fig=plt.figure(figsize=(10 , 10))
plt.subplot(2 , 2 , 1)
sns.boxplot(x='fueltype' , y='price' , data=df)
plt.subplot(2 , 2 , 2)
sns.boxplot(x='fueltype' , y='horsepower' , data=df)
plt.subplot(2 , 2 , 3)
sns.boxplot(x='fueltype' , y='carlength' , data=df)
plt.subplot(2 , 2 , 4)
sns.boxplot(x='fueltype' , y='carwidth' , data=df)
plt.tight_layout()


# In[15]:


fig=plt.figure(figsize=(10,10))
plt.subplot(2,  2 ,1)
sns.barplot(x='enginetype' , y='price' , data=df)
plt.subplot(2,  2 ,2)
sns.barplot(x='enginetype' , y='horsepower' , data=df)
plt.tight_layout()


# In[16]:


x = df.iloc[:,0:13].values
y = df.iloc[:,-1].values


# In[17]:


#data visualization 
plt.title('CAR PRICE DISTRIBUTION PLOT')
sns.histplot(df['price'] ,kde=True)


# In[18]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,2)
plt.title('CAR PRICE SPREAD')
sns.boxplot(y=df.price)


# In[19]:


#manual encoding
df.replace({'fueltype':{'gas':0, 'diesel':1}}, inplace=True)
#one hot encoding
df = pd.get_dummies(df, columns=['aspiration', 'enginelocation'], drop_first=True)


# In[20]:


plt.figure(figsize=(20,15))
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation between the columns')
plt.show()  


# In[21]:


fig=plt.figure(figsize=(10,5))
plt.title('Correlation between CityMpg and HighwayMpg')
sns.regplot(x='citympg', y='highwaympg', data=df)


# # Import train test split

# In[22]:


df.columns


# In[23]:


numerics = ['int16', 'int32', 'int64' , 'float16' ,'float32' , 'float64' ]
dfnum = df.select_dtypes(include=numerics)


# In[24]:


dfnum.head()


# In[25]:


temp=dfnum.iloc[: , 3:-1]
temp


# In[26]:


dfnum.head()
X=dfnum.iloc[: , 3:-1]
Y=dfnum.iloc[: , -1]


# In[27]:


X.head()


# In[28]:


Y.head()


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train , X_test , Y_train , Y_test = train_test_split(X  , Y , test_size=0.3 , random_state=101)


# In[31]:


X_train.shape


# In[32]:


X_test.shape


# # Linear Regression

# In[33]:


from sklearn.linear_model import LinearRegression


# In[34]:


model=LinearRegression()


# # Training model

# In[35]:


model.fit(X_train , Y_train)


# # Making Prediction

# In[36]:


y_pred=model.predict(X_test)


# # Checking accuracy of model

# In[37]:


from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score


# In[38]:


mse=mean_squared_error(Y_test , y_pred)
mae=mean_absolute_error(Y_test , y_pred)
r2=r2_score(Y_test  , y_pred)

print("Mean squared error is {}".format(mse))
print("Mean absolute error is {}".format(mae))
print("r2 score : {}".format(r2))


# # Prediction the output for users input

# In[39]:


import warnings
warnings.filterwarnings('ignore')

wheelbase=float(input("Enter the wheelbase : "))
carlength=float(input("Enter the carlength : "))
carwidth=float(input("Enter the carwidth : "))	
carheight=float(input("Enter the carheight : "))
curbweight=float(input("Enter the carweight : "))	
enginesize=float(input("Enter the enginesize : "))	
boreratio=float(input("Enter the boreratio : "))	
stroke=float(input("Enter the stroke : "))	
compressionratio=float(input("Enter the compressionratio : "))	
horsepower=float(input("Enter the horsepower : "))	
peakrpm=float(input("Enter the peakrpm : "))
citympg=float(input("Enter the citympg : "))
highwaympg=float(input("Enter the highwaympg : "))

ls=[wheelbase , carlength , carwidth , carheight ,curbweight ,enginesize,boreratio, stroke , compressionratio , horsepower ,peakrpm ,citympg ,highwaympg ]
arr=np.array(ls)
arr=arr.reshape(1 , -1)

result=model.predict(arr)
print("Price of the car is : {}".format(result))


# ### 
