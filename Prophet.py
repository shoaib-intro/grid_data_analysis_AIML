
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 20:04:22 2021

@author: ashoaib
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

col = ['Time','AC Primary Load','Grid Purchases','AC Operating Capacity','AC Required Operating Capacity']
df = pd.read_csv('DATA SET.csv')
cols = df.columns 
df = df[col]
df.set_index(pd.to_datetime(df['Time']), inplace=True)

# takes mean data as origional data has noise
mean_data = df.groupby(pd.Grouper(freq='1D')).mean()
data = pd.DataFrame(mean_data[col[1]])

data.plot(figsize=(16,4), title='AC Primary Load', legend=True, label='Grid Purchases')


#-------------------------Prophet

from prophet import Prophet

# AC Primary Load
data = pd.DataFrame(mean_data[col[1]])
data.reset_index(inplace=True)
data.columns = ['ds', 'y']
model = Prophet(changepoint_range=0.5, weekly_seasonality=False, yearly_seasonality=True, daily_seasonality=False)
model = model.fit(data)
future = model.make_future_dataframe(periods=len(data), freq='d', include_history =True)
forecast_1 = model.predict(future)
AC_primary_load = model


# Grid Purcahses
data = pd.DataFrame(mean_data[col[2]])
data.reset_index(inplace=True)
data.columns = ['ds', 'y']
model = Prophet(changepoint_range=0.5, weekly_seasonality=False, yearly_seasonality=True, daily_seasonality=False)
model = model.fit(data)
future = model.make_future_dataframe(periods=len(data), freq='d', include_history =True)
forecast_2 = model.predict(future)
Grid_Purcahses = model

# AC Operating Capacity
data = pd.DataFrame(mean_data[col[3]])
data.reset_index(inplace=True)
data.columns = ['ds', 'y']
model = Prophet(changepoint_range=0.5, weekly_seasonality=False, yearly_seasonality=True, daily_seasonality=False)
model = model.fit(data)
future = model.make_future_dataframe(periods=len(data), freq='d', include_history =True)
forecast_3 = model.predict(future)
AC_oc = model

# AC Required Operating Capacity
data = pd.DataFrame(mean_data[col[4]])
data.reset_index(inplace=True)
data.columns = ['ds', 'y']
model = Prophet(changepoint_range=0.5, weekly_seasonality=False, yearly_seasonality=True, daily_seasonality=False)
model = model.fit(data)
future = model.make_future_dataframe(periods=len(data), freq='d', include_history =True)
forecast_4 = model.predict(future)
AC_required_oc = model



# ploting

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,8))

AC_primary_load.plot(forecast_1, ax=axes[0,0], xlabel='2021-2022 time period', ylabel='AC Primary Load Frequency')
axes[0,0].set_title('AC Primary Load', fontsize=18, color='g')

Grid_Purcahses.plot(forecast_2, ax=axes[0,1], xlabel='2021-2022 time period', ylabel='Grid Purcahses Frequency')
axes[0,1].set_title('Grid Purcahses', fontsize=18, color='r')

AC_Operating_Capacity.plot(forecast_3, ax=axes[1,0], xlabel='2021-2022 time period', ylabel='AC Operating Capacity Frequency')
axes[1,0].set_title('AC Operating Capacity', fontsize=18, color='b')

AC_required_oc.plot(forecast_4, ax=axes[1,1], xlabel='2021-2022 time period', ylabel='AC Required Operating Capacity Frequency')
axes[1,1].set_title('AC Required Operating Capacity', fontsize=18, color='orange')

plt.suptitle('Grid Prediction using Prophet', fontsize=22, color='Orange', fontfamily='arial')
plt.legend(loc='best')
plt.tight_layout()



'''

#-------------------- RNN(LSTM)

# LSTM internal weights functions are sensitive to data
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#data = pd.DataFrame(mean_data[col[1]])
train, test = data[0:size], data[size:len(data)]

scaler = StandardScaler().fit(train[['y']])
train= scaler.transform(train[['y']])
test = scaler.transform(test[['y']])

TIME_STAMP = 6

# Split into chunks
def create_seq(series):
    x = []
    y = []
    for index in range(TIME_STAMP, len(series)):
        x.append(series[index-TIME_STAMP: index])
        y.append(series[index:index+1])
    return np.array(x),np.array(y)

x_train, y_train = create_seq(train)
x_test, y_test = create_seq(test)

# model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# Build the LSTM model from Sequential API
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


# Lets predict with the model
y_pred = model.predict(x_test)

# invert predictions
y_pred = scaler.inverse_transform(y_pred)
y_true = scaler.inverse_transform(y_test.reshape(-1,1))

# Get the root mean squared error (RMSE)
error = np.sqrt(np.mean((y_pred - y_true)**2))
print(error)




#--------------------- Performance Measures 
#only supported with categorical variables 

y_pred = y_pred.apply(lambda x: 0 if x < y_true.median() else 1)
y_true = y_true.apply(lambda x: 0 if x < y_true.median() else 1)

from sklearn.metrics import accuracy_score, recall_score, precision_score,f1_score, cohen_kappa_score, roc_auc_score
# accuracy: tp + tn / p + n
accuracy = accuracy_score(y_true,y_pred)
print('Accuracry Score {}'.format(accuracy))

# precison: tp / tp + fp
precision = precision_score(y_true,y_pred)
print('precion {}'.format(precision))

# recall: tp / tp + fn
recall = recall_score(y_true,y_pred)
print('recall {}'.format(recall))

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_true, y_pred)
print('F1 Score {}'.format(f1))

# kappa
kappa = cohen_kappa_score(y_true, y_pred)
print('Cohen Kappa {}'.format(kappa))

# ROC AUC
roc_auc = roc_auc_score(y_true, y_pred)
print('ROC AUC {}'.format(roc_auc))

'''
