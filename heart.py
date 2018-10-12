
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

train_values = pd.read_csv("C:\\Users\\Ayush\\Desktop\\Data\\heart\\train_values.csv")
train_labels = pd.read_csv("C:\\Users\\Ayush\\Desktop\\Data\\heart\\train_labels.csv")
test_values = pd.read_csv("C:\\Users\\Ayush\\Desktop\\Data\\heart\\test_values.csv")


# In[3]:


train_labels.head()


# In[142]:


a = train_values['patient_id'].unique()
a = list(a)


# In[143]:


for i in range(0,len(a)):
    if train_values['patient_id'][i] in a:
        train_values['patient_id'][i] = str(a.index(train_values['patient_id'][i]))  


# In[144]:


b = test_values['patient_id'].unique()
b = list(b)


# In[145]:


for i in range(0,len(b)):
    if test_values['patient_id'][i] in b:
        test_values['patient_id'][i] = str(b.index(test_values['patient_id'][i]))  


# In[146]:


c = train_values['thal'].unique()
c = list(c)


# In[147]:


for i in range(0,180):
    if train_values['thal'][i] in c:
        train_values['thal'][i] = (c.index(train_values['thal'][i]))  


# In[148]:


d = test_values['thal'].unique()
d = list(d)


# In[149]:


for i in range(0,90):
    if test_values['thal'][i] in d:
        test_values['thal'][i] = (d.index(test_values['thal'][i]))  


# In[150]:


train_labels = pd.DataFrame(train_labels['heart_disease_present'])


# In[151]:


train_values.head()


# In[152]:


def outlier25(a):
    b = np.array(a)
    b = np.percentile(b, 25)
    return b
def outlier75(a):
    b = np.array(a)
    b = np.percentile(b, 75)
    return b
def remove_outlier(a):
    outlier_index = []
    for i in range(0,len(a)):
        IQR = outlier75(a) - outlier25(a)
        if a[i] > outlier75(a)+1.5*IQR or a[i] < outlier25(a)-1.5*IQR:
            outlier_index.append(i)


# In[153]:


l = list(train_values)
print(l)
for i in range(1,len(l)):
    remove_outlier(train_values[str(l[i])])


# In[154]:


print(outlier_index)
#No outlier present in the entire data


# In[155]:


from sklearn import preprocessing
x = train_values.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
train_values = pd.DataFrame(x_scaled)
train_values.head()


# In[156]:


from sklearn import preprocessing
y = test_values.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
y_scaled = min_max_scaler.fit_transform(y)
test_values = pd.DataFrame(y_scaled)
test_values.head()


# In[157]:


for i in range(0,14):
    std = train_values[i].values.std()
    mean = train_values[i].values.mean()
    for j in range(0,len(train_values[i])):
        train_values[i][j] = (train_values[i][j] - mean) / std
train_values.head()


# In[158]:


for i in range(0,14):
    std = test_values[i].values.std()
    mean = test_values[i].values.mean()
    for j in range(0,len(test_values[i])):
        test_values[i][j] = (test_values[i][j] - mean) / std
test_values.head()


# In[159]:


from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(train_values, train_labels)


# In[160]:


y = clf.predict(test_values)
y = list(y)


# In[161]:


test_values = pd.read_csv("C:\\Users\\Ayush\\Desktop\\Data\\heart\\test_values.csv")
test_labels = pd.DataFrame(test_values['patient_id'])
test_labels['heart_disease_present'] = y
test_labels['heart_disease_present'] = test_labels['heart_disease_present'].astype(float)
test_labels.head()


# In[162]:


test_labels.to_csv("C:\\Users\\Ayush\\Desktop\\Data\\heart\\result4.csv", index=False)

