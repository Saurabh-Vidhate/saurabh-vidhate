#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import tree
import pickle
import joblib


# In[2]:


Heart=pd.read_csv("E:\project ML\Heart.csv")
Heart


# In[3]:


le=LabelEncoder()
le=le.fit(Heart.Sex)
Heart['Sex']=le.transform(Heart.Sex)


# In[4]:


le=le.fit(Heart.ChestPainType)
Heart['ChestPainType']=le.transform(Heart.ChestPainType)


# In[5]:


le=le.fit(Heart.RestingECG)
Heart['RestingECG']=le.transform(Heart.RestingECG)


# In[6]:


le=le.fit(Heart.ExerciseAngina)
Heart['ExerciseAngina']=le.transform(Heart.ExerciseAngina)


# In[7]:


le=le.fit(Heart.ST_Slope)
Heart['ST_Slope']=le.transform(Heart.ST_Slope)


# In[8]:


Heart


# In[ ]:





# In[9]:


#from sklearn.model_selection import train_test_split
predictors = Heart.drop("HeartDisease",axis=1)
HeartDisease = Heart["HeartDisease"]
X_train,X_test,Y_train,Y_test =train_test_split(predictors,HeartDisease,test_size=0.33,random_state=0)


# # Model Fitting

# Decision Tree Algorithm

# In[10]:


Tree=DecisionTreeClassifier(criterion='entropy',random_state=0,max_depth=4)
#lets predict it
Tree.fit(X_train,Y_train)


# In[11]:


y_pred=Tree.predict(X_test)
#from sklearn import metrics
#import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ",metrics.accuracy_score(Y_test,y_pred))


# In[12]:


filename = "Tree_model.sav"
pickle.dump(Tree,open(filename,'wb'))


# In[13]:


#loading model
load_model = pickle.load(open("Tree_model.sav",'rb'))


# In[17]:


input_data = [40,1,1,140,289,0,1,172,0,0.2,2]

#changing input data to numpy array
input_data_as_array = np.asarray(input_data)

#reshape array
input_data_reshape = input_data_as_array.reshape(1,-1)

prediction = load_model.predict(input_data_reshape)
print(prediction)

if (prediction[0] == 0):
    print('The person is does not have heart diisease')
else:
    print('The person have heart diisease')

