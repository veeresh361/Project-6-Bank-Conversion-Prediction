#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Reading the data

# In[66]:


file_name = "https://raw.githubusercontent.com/rajeevratan84/datascienceforbusiness/master/bank-additional-full.csv"
d = pd.read_csv(file_name, sep=';',nrows=5000)


# In[67]:


d.head(4)


# # Checking for missing Values

# In[4]:


d.isnull().sum()


# In[5]:


d.head(4)


# In[68]:


d['converted']=d['y'].apply(lambda x:1 if x=='yes' else 0)


# # Data Cleaning

# In[69]:


d.head()


# In[70]:


l=list(d['day_of_week'].unique())


# In[71]:


d['day_of_week']=d['day_of_week'].apply(lambda x: l.index(x)+1)


# In[72]:


d.head()


# In[73]:


cat_feat=[i for i in d.columns if d[i].dtype=='O']
cont_feat=[i for i in d.columns if d[i].dtype!='O']


# In[74]:


d['y'].value_counts()


# # Data Visualization

# In[22]:


fig=d.groupby('education')['y'].count().plot(kind='bar',grid=True,figsize=(12,8))
fig.set_ylabel('Count')
plt.show()


# In[23]:


d.head()


# In[27]:


d.groupby(['marital','housing'])['y'].count().unstack('housing').plot(kind='bar',figsize=(12,8))


# In[28]:


d.head(3)


# In[29]:


d['job'].value_counts()


# In[36]:


temp=d.groupby('job')['job'].count()/len(d)


# In[39]:


fig=temp.plot(kind='bar')
fig.axhline(y=0.05,color='r')


# # Rare Encoding

# In[75]:


def NON_rare(x_train,variable,tolerance):
    temp=x_train.groupby(variable)[variable].count()/len(x_train)
    non_freq=temp.loc[temp>tolerance].index.values
    return non_freq


# In[76]:


temp[temp>0.05].index.values


# In[77]:


x=d.drop(columns=['y'])
y=d['y']


# In[78]:


from sklearn.model_selection import train_test_split


# In[79]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)


# In[80]:


x_train


# In[81]:


def Rare_encoder(x_train,x_test,variable,tolerance):
    non_fq=NON_rare(x_train,variable,tolerance)
    x_train[variable]=np.where(x_train[variable].isin(non_fq),x_train[variable],'Rare')
    x_test[variable]=np.where(x_test[variable].isin(non_fq),x_test[variable],'Rare')
    return x_train,x_test


# In[83]:


x_train,x_test=Rare_encoder(x_train,x_test,'job',tolerance=0.05)


# In[65]:


d.groupby('education')['y'].sum()


# In[87]:


conversion_by_education=d.pivot_table(index='education',columns='converted',values='y',aggfunc=len)


# In[88]:


conversion_by_education


# In[90]:


conversion_by_education.columns=['Non_conversion','Conversion']


# In[94]:


conversion_by_education.plot(kind='pie',figsize=(12,8),startangle=90,subplots=True,autopct=lambda x: '%0.2f%%' % x,legend=False)


# In[99]:


conversion_by_job=d.pivot_table(index='job',columns='converted',values='y',aggfunc=len)


# In[100]:


conversion_by_job.columns=['Non_conversion','Conversion']


# In[105]:


conversion_by_job.plot(kind='pie',startangle=90,figsize=(20,10),autopct=lambda x: '%0.2f%%' % x,legend=False,subplots=True)


# In[106]:


d.head(4)


# # Encoding the Variable

# In[107]:


from feature_engine.categorical_encoders import CountFrequencyCategoricalEncoder


# In[125]:


cs=CountFrequencyCategoricalEncoder(encoding_method='count',variables=cat_feat)


# In[126]:


x_train=cs.fit_transform(x_train)
x_test=cs.transform(x_test)


# In[110]:


x_train


# In[111]:


cat_feat


# In[112]:


cat_feat=[i for i in x_train.columns if x_train[i].dtype=='O']


# In[113]:


cat_feat


# In[119]:


from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()


# In[116]:


x_train.drop(columns=['converted'],inplace=True)
x_test.drop(columns=['converted'],inplace=True)


# In[120]:


y_train=enc.fit_transform(y_train)
y_test=enc.transform(y_test)


# # Using KNN Classifier

# In[121]:


from sklearn.neighbors import KNeighborsClassifier


# In[129]:


error_rate=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    error_rate.append(np.mean(y_pred!=y_test))


# # Getting the accurate k

# In[131]:


plt.figure(figsize=(12,8))
plt.plot(range(1,40),error_rate,linestyle='dashed',color='r',markersize=10,marker='o')
plt.xlabel('K')
plt.ylabel('Error_rate')
plt.show()


# In[132]:


knn=KNeighborsClassifier(n_neighbors=15)
knn.fit(x_train,y_train)


# In[133]:


y_pred=knn.predict(x_test)


# In[136]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[135]:


accuracy_score(y_pred,y_test)


# In[137]:


print(classification_report(y_test,y_pred))


# In[ ]:




