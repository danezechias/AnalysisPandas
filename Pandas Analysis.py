#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 100)

# Data Import and Processing

etfs = 'socially_responsible_etfs.csv'
df = pd.read_csv(etfs, na_values='--')
# Displaying the first 5 rows
print(df.head())


# In[6]:


# Converting AUM column to floating point values (in millions)
df['AUM'] = df['AUM'].apply(lambda x: float(x[1:-1]) * (1000 if x[-1] == 'B' else 1))

df.head()


# In[7]:


# Spliting 'SEGMENT' column into individual columns using vectorized string method: extract
# the two formats (format1|format2) are: 
# format1: class1: market1 - segment1
# format2: class2: market2 segment2
temp_df = df['SEGMENT'].str.extract(r'(.+):\s+(.+)\s+-\s+(.+)|(.+):\s+(.+?)\s+(.+)')
temp_df.columns = ['Class1', 'Market1', 'Segment1', 'Class2', 'Market2', 'Segment2']

temp_df.head(10)


# In[8]:


# Updating the data frame with separate asset class, market, and segment information
df.insert(7, 'Class', np.where(temp_df['Class1'].notnull(), temp_df['Class1'], temp_df['Class2']))
df.insert(8, 'Market', np.where(temp_df['Market1'].notnull(), temp_df['Market1'], temp_df['Market2']))
df.insert(9, 'Segment', np.where(temp_df['Segment1'].notnull(), temp_df['Segment1'], temp_df['Segment2']))

# the original SEGMENT column
df = df.drop(columns=['SEGMENT'])

df.head()


# In[9]:


# Indexing and displaying the final data frame
df = df.set_index(['ISSUER', 'TICKER']).sort_index()
df.head()


# In[10]:


#Descriptive Analysis section


# In[22]:


# Computing the number of socially responsible ETFs from each issuer
issuer_counts = df.groupby('ISSUER')['FUND NAME'].count().sort_values(ascending=False)
print(issuer_counts)


# In[11]:


# Summarizing the total market capabilization of assets by issuer
issuer_aum = df.groupby('ISSUER')['AUM'].sum().sort_values(ascending=False)
print(issuer_aum)


# In[25]:


# Displaying the top 10 ETFs (by AUM)
top_10_etfs = df.sort_values(by='AUM', ascending=False).head(10)
print(top_10_etfs[['FUND NAME', 'AUM', 'EXPENSE RATIO', '3-MO TR', 'Class', 'Market', 'Segment']])


# In[26]:


# Summarizing the total market capabilization of assets by issuer
average_expense_ratio = df.groupby('ISSUER')['EXPENSE RATIO'].mean()
print(average_expense_ratio.sort_values())


# In[12]:


# Analyzing the distribution of asset classes
class_distribution = df['Class'].value_counts(normalize=True)
print(class_distribution)


# In[13]:


# Analyzing the distribution of markets
market_distribution = df['Market'].value_counts(normalize=True)
print(market_distribution)


# In[14]:


# Analyzing the distribution of segments 
segment_distribution = df['Segment'].value_counts(normalize=True)
print(segment_distribution)


# In[15]:


# Computing the mean 3-month total return as a function of asset class
average_returns = df.groupby('Class')['3-MO TR'].mean()
print(average_returns)


# In[16]:


# Computing the mean 3-month total return as a function of asset class and market
mean_returns_by_class_market = df.groupby(['Class', 'Market'])['3-MO TR'].mean()
print(mean_returns_by_class_market)


# In[17]:


# Computing the mean 3-month total return as a function of asset class, market, and segment
mean_returns_by_class_market_segment = df.groupby(['Class', 'Market', 'Segment'])['3-MO TR'].mean()
print(mean_returns_by_class_market_segment)


# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns

average_expense_ratio = df.groupby('ISSUER')['EXPENSE RATIO'].mean()

plt.figure(figsize=(10, 6))
average_expense_ratio.sort_values().plot(kind='bar')
plt.title('Average Expense Ratio by Issuer')
plt.xlabel('Issuer')
plt.ylabel('Average Expense Ratio')
plt.show()


# In[19]:


plt.figure(figsize=(6, 6))
df['Class'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Asset Class Distribution')
plt.show()


# In[20]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Class', y='3-MO TR', data=df)
plt.title('Distribution of 3-Month Total Return by Asset Class')
plt.xlabel('Asset Class')
plt.ylabel('3-Month Total Return')
plt.show()


# In[21]:


mean_returns_by_class_market = df.groupby(['Class', 'Market'])['3-MO TR'].mean().unstack()
plt.figure(figsize=(10, 6))
sns.heatmap(mean_returns_by_class_market, annot=True, cmap='coolwarm')
plt.title('Mean 3-Month Total Return by Asset Class and Market')
plt.xlabel('Market')
plt.ylabel('Asset Class')
plt.show()


# In[ ]:




