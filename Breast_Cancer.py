
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')
df.head(10)

df.columns

# count the number of rows and columns in dataset:
df.shape

sns.pairplot(df,hue = 'diagnosis', palette= 'coolwarm', vars = ['radius_mean', 'texture_mean', 'perimeter_mean','area_mean','smoothness_mean'])

# count the number of empty values in each columns:
df.isna().sum()
# drop the columns with all the missing values:
df = df.dropna(axis = 1)
df.shape
# Get the count of the number of Malignant(M) or Benign(B) cells
df['diagnosis'].value_counts()

# visualize the count:
sns.countplot(df['diagnosis'], label = 'count')

df.dtypes

# Rename the diagnosis data to labels:
df = df.rename(columns = {'diagnosis' : 'label'})
print(df.dtypes)

# define the dependent variable that need to predict(label)
y = df['label'].values
print(np.unique(y))

# Encoding categorical data from text(B and M) to integers (0 and 1)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(y) # M = 1 and B = 0
print(np.unique(Y))

# define x and normalize / scale value:
# define the independent variables, Drop label and ID, and normalize other data:
X  = df.drop(labels=['label','id'],axis = 1)
#scale / normalize the values to bring them into similar range:
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
print(X)

# Split data into training and testing data to verify accuracy after fitting the model
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.25, random_state=42)
print('Shape of training data is: ', x_train.shape)
print('Shape of testing data is: ', x_test.shape)

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
model = Sequential()
model.add(Dense(128, input_dim=30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])
model.summary()

# fit with no early stopping or other callbacks:
history = model.fit(x_train,y_train,verbose = 1,epochs = 100, batch_size = 64,validation_data = (x_test,y_test))

# plot the training and validation accuracy and loss at each epochs:
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)
plt.plot(epochs,loss,'y',label = 'Training loss')
plt.plot(epochs,val_loss,'r',label = 'Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs,acc,'y',label = 'Training acc')
plt.plot(epochs,val_acc,'r',label = 'Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predicting the Test set results:
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot = True)
print('Predictive accuracy: ', accuracy_score(y_test, y_pred)*100)

"""Logistic Regresion"""

#Fitting Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(random_state=0)
lg.fit(x_train, y_train)

# Making the Confusion Matrix:
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = lg.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot = True)
print('Predictive accuracy: ', accuracy_score(y_test, y_pred)*100)

"""Decision Tree

"""

#Fitting Decision Tree classifier to the training set
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt.fit(x_train, y_train)

# Making the Confusion Matrix:
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = dt.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot = True)
print('Predictive accuracy: ', accuracy_score(y_test, y_pred)*100)

"""Random Forest Classifier"""

# Fitting Decision Tree classifier to the training set:
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators= 10, criterion="entropy")
rfc.fit(x_train, y_train)

# Making the Confusion Matrix:
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = rfc.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True)
print('Predictive accuracy: ', accuracy_score(y_test, y_pred)*100)
