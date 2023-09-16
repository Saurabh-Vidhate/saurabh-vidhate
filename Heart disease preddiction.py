#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas


# In[2]:


pip install category_encoders


# In[3]:


pip install sklearn


# In[4]:


pip install seaborn


# In[2]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import SVC
from sklearn import svm
from matplotlib.cm import rainbow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB 
from sklearn.naive_bayes import BernoulliNB


# In[6]:


Heart=pd.read_csv("D:\project ML\Heart.csv")
Heart


# In[7]:


Heart.info()


# In[8]:


Heart.describe()


# In[9]:


S=Heart['Sex']
S


# In[10]:


#from sklearn.preprocessing import LabelEncoder


# In[11]:


le=LabelEncoder()
le=le.fit(Heart.Sex)
Heart['Sex']=le.transform(Heart.Sex)


# In[12]:


le=le.fit(Heart.ChestPainType)
Heart['ChestPainType']=le.transform(Heart.ChestPainType)


# In[13]:


le=le.fit(Heart.RestingECG)
Heart['RestingECG']=le.transform(Heart.RestingECG)


# In[14]:


le=le.fit(Heart.ExerciseAngina)
Heart['ExerciseAngina']=le.transform(Heart.ExerciseAngina)


# In[15]:


le=le.fit(Heart.ST_Slope)
Heart['ST_Slope']=le.transform(Heart.ST_Slope)


# In[16]:


Heart


# In[17]:


Heart.to_excel('Heart1.xlsx')


# # Exploratory Data Analysis(EDA)

# In[18]:


heart.shape[1]


# In[19]:


heart.shape[:1]


# In[3]:


heart=pd.read_excel("D:\project ML\Heart1.xlsx")
heart


# In[21]:










e have the following encoders to given columns
Sex:
Male=['1']
Female=['0']

#ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]

ChestPainType:
    TA=['3']
    ATA=['1']
    NAP=['2']
    ASY=['0']

RestingECG: resting electrocardiogram results 
    [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), 
     LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria
   Normal=['1']
   ST=['2']
   LVH=['0']
     
ExerciseAngina:
     Yes(y)=['1']
     No(N)=['0']
     
ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
     Flat=['1']
     Up=['2']


# In[22]:


#import seaborn as sns


# In[23]:


y=heart['HeartDisease']
sns.countplot(y)
Heart_Disease=heart.HeartDisease.value_counts()
print('1 is patient have heart disease','\n','0 is patient doesnt have heart disease')
Heart_Disease


# We conclude here in our dataset heartdisease patient is more.

# In[24]:


sns.barplot(heart['Sex'],heart['HeartDisease'])


#  We notice that males aremore likely to have heart problems then female

# # Percentage of patients with or without heart disease

# In[ ]:



        


# In[25]:


sns.countplot(x='HeartDisease',data=heart,palette="bwr")
plt.show()


# In[26]:


countFemale=len(heart[heart.Sex==0])
countMale=len(heart[heart.Sex==1])
#print(countFemale/(len(heart[heart.Sex]))*100)
#print(countMale/(len(heart[heart.Sex]))*100)
print("Percentage of Female Patients:{:.2f}%".format((countFemale)/(len(heart.Sex))*100))
print("Percentage of Male Patients:{:.2f}%".format((countMale)/(len(heart.Sex))*100))


# In[27]:


countNoDisease = len(heart[heart.HeartDisease == 0])
countHaveDisease = len(heart[heart.HeartDisease == 1])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(heart.HeartDisease))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(heart.HeartDisease))*100)))


# In[28]:


heart.groupby('HeartDisease').mean()


# # Heart Disease Frequency for ages

# In[ ]:





# In[30]:


pd.crosstab(heart.Age,heart.HeartDisease).plot(kind='bar',figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()


# In above Bar diagram the frequency is varying and increasing as the age is increases at upto 55-65 and in that male have
# a higher frequency than female after the age of 65 the frequency decreases and in this female have 
# a higher frequency than a male 
# 

# # Heart Disease Frequency for Male and Female

# In[392]:


pd.crosstab(heart.Sex,heart.HeartDisease).plot(kind="bar",figsize=(11,4))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Sex(0=Female,1=male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease","Have Disease"])
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndSex.png')
plt.show()


# In above Heart Disease frequecy for male and female bar diagram female patients have low heartdisease frequency as compared
# to a male patients.We can notice here in feamle heartdisease patients is less in total patients and in male patients 
# it is viceversa

# # ChestPainType vs Cholestrol

# In[32]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='Cholesterol',y='ChestPainType',data=heart,hue='HeartDisease')
plt.show()


# #ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
# 
# ChestPainType:
#     TA=['3']
#     ATA=['1']
#     NAP=['2']
#     ASY=['0']
#     
# We plot a scatterplot above for Chestpaintype and Cholestrol level of patients.
# 

# In[33]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='ChestPainType',y='RestingBP',data=Heart,hue='HeartDisease')
plt.show()


# The Scatterrplot of ChestPainType vs RestingBP 

# # Age vs Maximum heart rate

# In[34]:


plt.figure(figsize=(10,5))
plt.scatter(x=heart.Age[heart.HeartDisease==1], y=heart.MaxHR[(heart.HeartDisease==1)], c="green")
plt.scatter(x=heart.Age[heart.HeartDisease==0], y=heart.MaxHR[(heart.HeartDisease==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()


# # Fasting Blood Sugar data

# In[35]:


pd.crosstab(heart.FastingBS,heart.HeartDisease).plot(kind="bar",figsize=(20,10),color=['#4286f4','#f49242'])
plt.title("Heart disease according to FBS")
plt.xlabel('FBS- (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation=90)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Disease or not')
plt.show()


# Fasting Blood Sugar which is greater than 120 mg/dl 
# in the patients, have a less frequency of having disease than a patients having lesser than 120mg/dl fasting blood sugar. 

# In[36]:


heart.isnull().sum()


# In[295]:


names=['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']


# In[299]:


#Set the width and height of the plot
f = plt.subplots(figsize=(7, 5))

#Correlation plot
df_corr = heart.loc[:,names]
#Generate correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, annot = True, cmap='coolwarm',linewidths=.1)
plt.show()


# NEGATIVE CORRELATION :
#     A negative, or inverse correlation, between two variables, 
# indicates that one variable increases while the other decreases, and vice-versa. 
# This relationship may or may not represent causation between the two variables, 
# but it does describe an observable pattern\
# 
# POSITIVE CORRELATION:
#     A positive correlation is a relationship between two variables that tend to move in the same direction.
#     A positive correlation exists when one variable tends to decrease as the other variable decreases,
#     or one variable tends to increase when the other increases.\
# 
# ZERO CORRELATION:
#      A value of zero indicates no relationship between the two variables being compared.

# In[39]:


#POSITIVE CORRELATION
plt.figure(figsize=(10,5))
plt.scatter(x=heart['Age'], y=heart['Oldpeak'], c="green")

plt.xlabel("Age")
plt.ylabel("Oldpeak")
plt.show()


# In[40]:


#NEGATIVE CORRELATION
plt.figure(figsize=(10,5))
plt.scatter(x=heart['Age'], y=heart['MaxHR'], c="blue")
plt.xlabel("Age")
plt.ylabel("MaxHR")
plt.show()


# In[41]:


#ZERO CORRELATION
plt.figure(figsize=(10,5))
plt.scatter(x=heart['Oldpeak'], y=heart['Cholesterol'], c="blue")
plt.xlabel("Oldpeak")
plt.ylabel("Cholesterol")
plt.show()


# In[42]:


names=['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope','HeartDisease']


# In[43]:


#Set the width and height of the plot
f, ax = plt.subplots(figsize=(9, 7))

#Correlation plot
df_corr = heart.loc[:,names]
#Generate correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, annot = True, cmap='coolwarm',linewidths=.1)
plt.show()


# 

# In[44]:


corr


# In[ ]:





# In[45]:


df_corr


# # Train Test Split

# In[369]:


#from sklearn.model_selection import train_test_split
predictors = heart.drop("HeartDisease",axis=1)
HeartDisease = heart["HeartDisease"]
X_train,X_test,Y_train,Y_test =train_test_split(predictors,HeartDisease,test_size=0.33,random_state=1)


# In[370]:


X_test.shape


# In[371]:


X_train.shape


# In[372]:


X_test


# In[ ]:





# In[373]:


Y_train.shape


# In[374]:


Y_test


# In[375]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[376]:


#X_train=sc.fit_transform(X_train)
#X_test=sc.fit_transform(X_test)
#X_train.shape


# In[377]:


X_test


# In[ ]:





# # Model Fitting

# # Decision Tree Algorithm

# In[378]:


Tree=DecisionTreeClassifier(criterion='entropy',random_state=1,max_depth=4)
Tree


# In[379]:


#lets predict it
Tree.fit(X_train,Y_train)


# In[380]:


y_pred=Tree.predict(X_test)
print(y_pred)
y_pred.shape


# In[381]:


print(Y_test)


# In[382]:


#from sklearn import metrics
#import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ",accuracy_score(Y_test,y_pred))


# Hence we conclude here,there is 86.79% accuracy in the model which is good.Accuracy in the model defines\ 
# that predicted value is 86.79% accurate with respect to observed value

# Create the decision tree and visualize it

# In[383]:


conf=confusion_matrix(Y_test,y_pred)
print(conf)
f, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(conf/np.sum(conf),annot=True,cmap='winter',fmt='.2%',linewidths=0.3, linecolor='black')


# In[384]:


print(classification_report(Y_test,y_pred))


# In[385]:


from sklearn import tree
plt.figure(figsize=(35,10))
tree.plot_tree(Tree,filled=True, 
              rounded=True, 
              fontsize=14);


# In[386]:


#from sklearn.model_selection import train_test_split  
#from sklearn.tree import DecisionTreeClassifier


# In[387]:


y = heart['HeartDisease'] 
X = heart.drop(['HeartDisease'], axis = 1) 
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
dt_scores = [] 
for i in range(1, len(X.columns) + 1):     
    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 1)     
    dt_classifier.fit(X_train, Y_train) 
    dt_scores.append(dt_classifier.score(X_test, Y_test)) 
plt.figure(figsize=(12,7))
plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color = 'green') 
for i in range(1, len(X.columns) + 1): 
    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1])) 
plt.xticks([i for i in range(1, len(X.columns) + 1)]) 
plt.xlabel('Max features') 
plt.ylabel('Scores') 
plt.title('Decision Tree Classifier scores for different number of maximum features') 
plt.show()


# In[ ]:





# # Simple Linear Regression Model

# In[66]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression


# In[67]:


x_test=X_test['Oldpeak']
x_test.shape


# In[68]:


x_test.values.reshape(-1,1)
x_test


# In[69]:


x_train=X_train['Oldpeak']
x_train.shape


# In[ ]:





# In[ ]:





# In[72]:


X_train =pd.DataFrame(X_train)
X_train


# In[73]:


X_train.ndim

X_test=np.array(x_test)
X_test

# In[74]:


X_test=pd.DataFrame(x_test)
X_test


# In[75]:


X_test.ndim


# In[76]:


Y_test


# In[77]:


regr = linear_model.LinearRegression()
regr.fit(X_train,Y_train)


# In[78]:


#print(X_test)
arr=np.array(x_test)
x=arr.flatten()
x.shape


# In[79]:


#x_test=np.array(x_test).reshape(-1,1)
y_pred = regr.predict(X_test)


# In[80]:


y=np.array(Y_test)
y.shape
#np.array([y_pred]).shape


# In[81]:


regr.coef_


# In[82]:


regr.intercept_


# In[83]:


metrics.mean_absolute_error(Y_test,y_pred)


# In[84]:


metrics.mean_squared_error(Y_test,y_pred)


# In[85]:


metrics.r2_score(Y_test,y_pred)


# In[86]:


#plot outputs
plt.scatter(X_test,Y_test)
plt.plot(X_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# In[87]:


def estimate_coef(X, Y):
    # number of observations/points 
    n = np.size(X) 

    # mean of x and y vector 
    m_x, m_y = np.mean(X), np.mean(Y) 

    # calculating cross-deviation and deviation about x 
    #Cross-deviation = summation(y*x) - (num_of_obs * mean(y) * mean(x))
    SS_xy = np.sum(Y*X) - n*m_y*m_x 
    
    #Deviation about x = summation(x*x) - (num_of_obs * mean(x) * mean(x))
    SS_xx = np.sum(X*X) - n*m_x*m_x 

    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 

    return(b_0, b_1) 

def plot_regression_line(X, Y, b): 
    
     

    # predicted response vector 
    y_pred = b[0] + b[1]*X
    
    # plotting the actual points as scatter plot 
    plt.scatter(X,Y, color = "m",marker = "o", s = 30)

    # plotting the regression line 
    plt.plot(X, y_pred, color = "g") 

# putting labels 
    plt.xlabel('X') 
    plt.ylabel('Y') 

    # function to show plot 
    plt.show() 

def main(): 
    # observations 
    X=x_test        #print(X_test)  #arr=np.array(X_test)  #x=arr.flatten()
    Y=Y_test        #y=np.array(Y_test)

    # estimating coefficients 
    b = estimate_coef(X, Y) 
    print("Estimated coefficients:\nb_0 = {} nb_1 = {}".format(b[0], b[1])) 

# plotting regression line 
    plot_regression_line(X, Y, b) 

if __name__ == "__main__": 
    main() 


# In[244]:


predictors = heart.drop("HeartDisease",axis=1)
HeartDisease = heart["HeartDisease"]
X_train,X_test,Y_train,y_test =train_test_split(predictors,HeartDisease,test_size=0.33,random_state=1)


# In[245]:


classifier=svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)


# In[246]:


classifier.score(X_test,y_test)


# In[247]:


classifier=svm.SVC(kernel='poly')
classifier.fit(X_train,Y_train)
y_pred=classifier.predict(X_test)
classifier.score(X_test,y_test)


# In[248]:


classifier=svm.SVC(kernel='rbf')
classifier.fit(X_train,Y_train)
y_pred=classifier.predict(X_test)
classifier.score(X_test,y_test)


# In[249]:


Y_train.shape


# In[250]:


y_test.shape


# In[251]:


#from sklearn.svm import SVC
#from matplotlib.cm import rainbow
svc_scores = []
kernels = ['linear', 'poly', 'rbf']
y = heart['HeartDisease'] 
X = heart.drop(['HeartDisease'], axis = 1.0) 


for i in range(len(kernels)):     
    svc_classifier = SVC(kernel = kernels[i])     
    svc_classifier.fit(X_train, Y_train) 
    svc_scores.append(svc_classifier.score(X_test, y_test)) 
colors = rainbow(np.linspace(0, 1, len(kernels))) 
plt.bar(kernels, svc_scores, color = colors) 
for i in range(len(kernels)):     
    plt.text(i, svc_scores[i], svc_scores[i]) 
plt.xlabel('Kernels') 
plt.ylabel('Scores') 
plt.title('Support Vector Classifier scores for different kernels') 
plt.show()


# In[273]:


accuracy_score(y_test,Y_pred)


# In[302]:


cm=confusion_matrix(y_test,Y_pred)
cm


# In[313]:


f, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm/np.sum(cm),annot=True,cmap='winter',fmt='.2%',linewidths=0.3, linecolor='black')


# In[255]:


print(classification_report(y_test,Y_pred))


# In[91]:


heart


# # Logistic Regression

# In[7]:


heart.isna().sum()


# In[8]:


heart


# In[346]:


#from sklearn.model_selection import train_test_split
predictors = heart.drop("HeartDisease",axis=1)
HeartDisease = heart["HeartDisease"]
X_train,X_test,Y_train,Y_test =train_test_split(predictors,HeartDisease,test_size=0.33,random_state=1)


# In[347]:


lgr=LogisticRegression(max_iter=500)


# In[348]:


lgr.fit(X_train,Y_train)


# In[349]:


y_pred=lgr.predict(X_test)


# In[350]:


y_pred


# In[351]:


from sklearn.metrics import accuracy_score, classification_report


# In[352]:


accuracy_score(Y_test,y_pred)


# In[353]:


cfm=confusion_matrix(Y_test,y_pred)


# In[354]:


cfm


# In[355]:


f, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cfm/np.sum(cfm),annot=True,cmap='winter',fmt='.2%',linewidths=0.3, linecolor='black')


# In[359]:


print(classification_report(Y_test,y_pred))


# Recall is a metric that quantifies the number of correct positive predictions made \
# out of all positive predictions that could have been made
# 
# The precision measures the model's accuracy in classifying a sample as positive.
# 
# The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean. It is primarily used to compare the performance of two classifiers. Suppose that classifier A has a higher recall, and classifier B has higher precision.

# In[ ]:


#from sklearn.preprocessing import StandardScaler
#from sklearn.naive_bayes import GaussianNB 
#from sklearn.naive_bayes import BernoulliNB


# In[4]:


#extracting categorical attributes
x = heart.iloc[:,[1,2,5,6,8,10]].values  
y = heart.iloc[:, 11].values 

y=pd.DataFrame(y)

#splitting dataset for train and test 
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size = 0.33, random_state = 1)

#feature scaling
sc = StandardScaler()  
x_train = sc.fit_transform(x_train)  
x_test = sc.transform(x_test) 

# Fitting Naive Bayes to the Training set 
classifier = BernoulliNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results  
y_pred = classifier.predict(x_test) 
y_pred

y_pred=pd.DataFrame(y_pred)

#accuracy Score
accuracy_score(y_test,y_pred)

#confusion matrix for accuracy
cm = confusion_matrix(y_test, y_pred) 

f, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm/np.sum(cm),annot=True,cmap='winter',fmt='.2%',linewidths=0.3, linecolor='black')

print(classification_report(y_test,y_pred))


# In[5]:


#extracting categorical attributes
x = heart.iloc[:,[1,2,5,6,8,10]].values  
y = heart.iloc[:, 11].values 


# In[6]:


y=pd.DataFrame(y)


# In[7]:


#splitting dataset for train and test 
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size = 0.33, random_state = 1)


# In[8]:


#feature scaling
sc = StandardScaler()  
x_train = sc.fit_transform(x_train)  
x_test = sc.transform(x_test) 


# In[9]:


# Fitting Naive Bayes to the Training set 
classifier = BernoulliNB()
classifier.fit(x_train, y_train)


# In[10]:


# Predicting the Test set results  
y_pred = classifier.predict(x_test) 
y_pred


# In[11]:


y_pred=pd.DataFrame(y_pred)


# In[12]:


y_pred


# In[13]:


#accuracy Score
accuracy_score(y_test,y_pred)


# In[14]:


#confusion matrix for accuracy
cm = confusion_matrix(y_test, y_pred) 


# In[15]:


cm


# In[18]:


f, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm/np.sum(cm),annot=True,cmap='winter',fmt='.2%',linewidths=0.3, linecolor='black')


# In[20]:


print(classification_report(y_test,y_pred))


# In[ ]:




