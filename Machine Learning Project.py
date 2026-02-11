#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# In[4]:


#Loading dataset
df = pd.read_csv("hiring (1).csv")
print(df.head())


# In[5]:


# Convert word numbers in experience column to numeric
word_to_num = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10}
df['experience'] = df['experience'].replace(word_to_num)
df['experience'] = pd.to_numeric(df['experience'], errors='coerce')

# Fill missing values with median
df['experience'] = df['experience'].fillna(df['experience'].median())
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].median())

#Prepare features (X) and target (y)
X = df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']]
y = df['salary($)']

#Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

#Predict salaries for new candidates

# Candidate 1: 2 yrs experience, 9 test score, 6 interview score
salary1 = model.predict([[2, 9, 6]])

# Candidate 2: 12 yrs experience, 10 test score, 10 interview score
salary2 = model.predict([[12, 10, 10]])

#Print results
print("Predicted salary for candidate 1:", salary1[0])
print("Predicted salary for candidate 2:", salary2[0])


# In[ ]:




