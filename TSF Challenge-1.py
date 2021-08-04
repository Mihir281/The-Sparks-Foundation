#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# In[26]:


df=pd.read_csv("C:\\Users\\HP\\Dataset_1.csv")


# In[28]:


df.head()


# In[57]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("Hours Studied")
plt.ylabel("Student Scores")
plt.scatter(df.Hours,df.Scores)
plt.show()


# In[59]:


import seaborn as sns
sns.regplot(x=df["Hours"],y=df["Scores"])
plt.title("Hours vs Scores")


# In[49]:


x=df.iloc[:,:-1].values
y=df.iloc[:,1].values
train_x,val_x,train_y,val_y=train_test_split(x,y,random_state=0)


# In[53]:


#Regression Analysis
reg=linear_model.LinearRegression()
reg.fit(train_x,train_y)


# In[54]:


#To find the score for a student that studies for 9.25 hours/day.
hours=[9.25]
score=reg.predict([hours])
print(score)

