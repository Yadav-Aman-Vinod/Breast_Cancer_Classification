import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation

df = pd.read_csv('data.csv')

df = df.dropna(axis=1)

df = df.rename(columns={'diagnosis': 'label'})

labelencoder = LabelEncoder()
df['label'] = labelencoder.fit_transform(df['label'])

y = df['label'].values
X = df.drop(['label', 'id'], axis=1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = Sequential([
    Dense(128, input_dim=x_train.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_test, y_test), verbose=1)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

y_pred = (model.predict(x_test) > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
print('Neural Network Predictive Accuracy: ', accuracy_score(y_test, y_pred) * 100)

lg = LogisticRegression(random_state=0)
lg.fit(x_train, y_train)
y_pred_lg = lg.predict(x_test)
cm_lg = confusion_matrix(y_test, y_pred_lg)
sns.heatmap(cm_lg, annot=True)
print('Logistic Regression Predictive Accuracy: ', accuracy_score(y_test, y_pred_lg) * 100)

dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True)
print('Decision Tree Predictive Accuracy: ', accuracy_score(y_test, y_pred_dt) * 100)

rfc = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
rfc.fit(x_train, y_train)
y_pred_rfc = rfc.predict(x_test)
cm_rfc = confusion_matrix(y_test, y_pred_rfc)
sns.heatmap(cm_rfc, annot=True)
print('Random Forest Predictive Accuracy: ', accuracy_score(y_test, y_pred_rfc) * 100)
