#importing required libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
%matplotlib inline


df=pd.read_csv('heart.csv')
df.head(5)


# It's a clean, easy to understand set of data. However, the meaning of some of the column headers are not obvious. Here's what they mean,
# 
# 1.age: The person's age in years
# 
# 2.sex: The person's sex (1 = male, 0 = female)
# 
# 3.cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
# 
# 4.trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
# 
# 5.chol: The person's cholesterol measurement in mg/dl
# 
# 6.fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
# 
# 7.restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
# 
# 8.thalach: The person's maximum heart rate achieved
# 
# 9.exang: Exercise induced angina (1 = yes; 0 = no)
# 
# 10.oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
# 
# 11.slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
# 
# 12.ca: The number of major vessels (0-3)
# 
# 13.thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
# 
# 14.target: Heart disease (0 = no, 1 = yes)
# 

# In[4]:


# df.describe()


# In[5]:


# df.info()


df


# In[18]:


df.columns.values


# In[19]:


# chest_pain=pd.get_dummies(df['cp'],prefix='cp',drop_first=True)
# df=pd.concat([df,chest_pain],axis=1)
# df.drop(['cp'],axis=1,inplace=True)
# sp=pd.get_dummies(df['slope'],prefix='slope')
# th=pd.get_dummies(df['thal'],prefix='thal')
# rest_ecg=pd.get_dummies(df['restecg'],prefix='restecg')
# frames=[df,sp,th,rest_ecg]
# df=pd.concat(frames,axis=1)
# df.drop(['slope','thal','restecg'],axis=1,inplace=True)


# In[20]:


df.head(5)


# In[21]:


X = df.drop(['target'], axis = 1)
y = df.target.values


# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[23]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_test


# In[24]:


import sys
np.set_printoptions(threshold=sys.maxsize)
X_test


# ## Importing the Keras libraries and packages

# In[25]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras
from keras.models import Sequential
from keras.layers import Dense
import warnings



classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(keras.layers.Dense(13, activation = 'relu', kernel_initializer = 'uniform', input_dim = 13))

# Adding the second hidden layer
classifier.add(Dense(13, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# In[60]:


# Predicting the Test set results
X_user = pd.DataFrame(
    {'age': 75,
     'sex': 1,
     'cp': 0,
     'trestbps': 140,
     'chol': 192,
     'fbs': 0,
     'restecg': 1,
     'thalach': 148,
     'exang': 0,
     'oldpeak': 0.4,
     'slope': 1,
     'ca': 0,
     'thal': 1
    },index=[0])
# chest_pain=pd.get_dummies(X_user['cp'],prefix='cp',drop_first=True)
# X_user=pd.concat([X_user,chest_pain],axis=1)
# X_user.drop(['cp'],axis=1,inplace=True)
# sp=pd.get_dummies(X_user['slope'],prefix='slope')
# th=pd.get_dummies(X_user['thal'],prefix='thal')
# rest_ecg=pd.get_dummies(X_user['restecg'],prefix='restecg')
# frames=[X_user,sp,th,rest_ecg]
# X_user=pd.concat(frames,axis=1)
# X_user.drop(['slope','thal','restecg'],axis=1,inplace=True)
X_user = sc.transform(X_user)
y_pred = classifier.predict(X_user)
print(float(y_pred[0]))
print(int(y_pred[0][0]))


# In[29]:


# X_user = pd.DataFrame(
#     {'age': 57,
#      'sex': 1,
#      'cp': 0,
#      'trestbps': 140,
#      'chol': 192,
#      'fbs': 0,
#      'restecg': 1,
#      'thalach': 148,
#      'exang': 0,
#      'oldpeak': 0.4,
#      'slope': 1,
#      'ca': 0,
#      'thal': 1
#     },index=[0])
# y_test.shape


# In[44]:


import seaborn as sns
from sklearn.metrics import confusion_matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred.round())
sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",cbar=False)
# accuracy score

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred.round())
print('accuracy of the model: ',ac)


classifier.save(r'C:\Users\ishan\Desktop\SC')

