#!/usr/bin/env python
# coding: utf-8

# ![](IRIS.JPG)

# ![IRIS.JPG.png](attachment:IRIS.JPG.png)

# ![iris-flower.webp](attachment:iris-flower.webp)

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


df=pd.read_csv(r'C:\Users\ayushi\Downloads\archive\Iris.csv')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.shape


# # Cheking null values

# In[6]:


df.isna().sum()


# In[7]:


df['Species'].unique()


# # VISUALIZATION OF DATASET

# In[8]:


df=df.drop(['Id'],axis=1)


# In[9]:


sns.pairplot(df,hue='Species')


# In[10]:


sns.scatterplot(x='SepalWidthCm', y='SepalLengthCm', hue='Species', data=df)


# In[11]:


sns.histplot(df[df['Species']=='Iris-setosa']['PetalLengthCm'],color='red',kde=True,stat="density")
sns.histplot(df[df['Species']=='Iris-versicolor']['PetalLengthCm'],color='blue',kde=True,stat="density")
sns.histplot(df[df['Species']=='Iris-virginica']['PetalLengthCm'],color='green',kde=True,stat="density")


# In[12]:


sns.histplot(df[df['Species']=='Iris-setosa']['SepalLengthCm'],color='red',kde=True,stat="density")
sns.histplot(df[df['Species']=='Iris-versicolor']['SepalLengthCm'],color='blue',kde=True,stat="density")
sns.histplot(df[df['Species']=='Iris-virginica']['SepalLengthCm'],color='green',kde=True,stat="density")


# In[13]:


sns.histplot(df[df['Species']=='Iris-setosa']['PetalWidthCm'],color='red',kde=True,stat="density")
sns.histplot(df[df['Species']=='Iris-versicolor']['PetalWidthCm'],color='blue',kde=True,stat="density")
sns.histplot(df[df['Species']=='Iris-virginica']['PetalWidthCm'],color='green',kde=True,stat="density")


# In[14]:


sns.histplot(df[df['Species']=='Iris-setosa']['SepalWidthCm'],color='red',kde=True,stat="density")
sns.histplot(df[df['Species']=='Iris-versicolor']['SepalWidthCm'],color='blue',kde=True,stat="density")
sns.histplot(df[df['Species']=='Iris-virginica']['SepalWidthCm'],color='green',kde=True,stat="density")


# In[15]:


df1=df.copy()


# In[16]:


df1['PetalWidthCm'].unique()


# In[17]:


#Checking for outliers
sns.boxplot(df1['SepalWidthCm'])


# In[18]:


#Removing the outliers using the IQR method
Q1=df1['SepalWidthCm'].quantile(0.25)
Q3=df1['SepalWidthCm'].quantile(0.75)
IQR=Q3-Q1
ll=Q1-1.5*IQR
ul=Q3+1.5*IQR


# In[19]:


df1=df1[(df1['SepalWidthCm']<ul) & (df1['SepalWidthCm']>ll)]


# In[20]:


df1.shape


# In[21]:


df['Species'].unique()


# In[22]:


#Encoding the target variable using Label Encocoder
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1['Species']=le.fit_transform(df1['Species'])
df1.head()


# In[23]:


#Performing train test split
from sklearn.model_selection import train_test_split as tts
x=df1.drop(['Species'],axis=1)
y=df1['Species']
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2,random_state=42)


# In[24]:


y_train.value_counts()


# In[25]:


from sklearn.metrics import accuracy_score as ac, precision_score as pc,f1_score as f1


# # Logistic Regression

# In[26]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[27]:


lr.fit(x_train,y_train)
y_pred_lr=lr.predict(x_test)


# In[28]:


ac(y_pred_lr,y_test)


# # Modeling KNN

# In[29]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
y_pred_knn=knn.predict(x_test)
ac(y_pred_knn,y_test)


# # Categorical Naive Bayes

# In[30]:


from sklearn.naive_bayes import CategoricalNB
cnb=CategoricalNB()
cnb.fit(x_train,y_train)
y_pred_cnb=cnb.predict(x_test)
ac(y_pred_cnb,y_test)


# # Gaussian Naive Bayes

# In[31]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_pred_gnb=gnb.predict(x_test)
ac(y_pred_gnb,y_test)


# # KNN GIVES BEST ACCURACY FOR THE IRIS DATASET

# We can see that accuracy of the model is 96.66 percent which is very accurate

# In[ ]:




