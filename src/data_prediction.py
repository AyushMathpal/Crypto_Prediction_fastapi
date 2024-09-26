import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler,StandardScaler
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import LSTM
from keras.losses import Huber
from keras.layers import Dense, Dropout,BatchNormalization
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping
from keras.metrics import BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall, AUC, binary_accuracy
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras.layers import RepeatVector
from keras.layers import TimeDistributed



#############################################Functions#############################################

def split_sequences(sequences, n_steps_in, n_steps_out):
 X, y = list(), list()
 for i in range(len(sequences)):

 # find the end of this pattern
  end_ix = i + n_steps_in
  out_end_ix = end_ix + n_steps_out
  # check if we are beyond the dataset
  if out_end_ix > len(sequences):
    break
  seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix,:]
  X.append(seq_x)
  y.append(seq_y)

 # gather input and output parts of the pattern

 return np.array(X), np.array(y)


# def build_model():
#     model = Sequential()
#     model.add(LSTM(256, input_shape=(X_train.shape[1:]), return_sequences=True))
#     model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

#     model.add(LSTM(128, return_sequences=True))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.1))

#     model.add(LSTM(64))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.2))

#     model.add(Dense(32, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.2))

#     model.add(Dense(1, activation='tanh'))

#     opt = tf.keras.optimizers.RMSprop(lr=7e-3)
#     model.compile(loss='mse',optimizer=opt, metrics=['mse'])
#     return model

def scheduler(epoch, lr):
  if epoch < 8:
    return lr
  else:
    return float(lr * tf.math.exp(-0.2))
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
#############################################Calls#############################################

df=pd.read_csv("src/Final_Dataset.csv",index_col=0, parse_dates=True)
df.drop("Volume",axis=1,inplace=True)

scaler = StandardScaler()
scaled_data=scaler.fit_transform(df)

n_future = 30   # Number of days we want to look into the future based on the past days.
n_past = 60     # Number of past days we want to use to predict the future.

train_size = int(len(scaled_data) * 0.7)
train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]
X_train, y_train = split_sequences(train_data,n_past,n_future)
X_test, y_test = split_sequences(test_data,n_past,n_future)
n_features=X_train.shape[2]
n_output_features=y_train.shape[2]



es_callback=EarlyStopping(
    monitor='val_loss',min_delta=0,patience=10,verbose=1,mode='auto'
    ,baseline=None,restore_best_weights=True
)

opt = Adam(learning_rate=0.0001)
# define model
encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(200,return_sequences = True, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)
encoder_states1 = encoder_outputs1[1:]
encoder_l2 = tf.keras.layers.LSTM(200, return_state=True)
encoder_outputs2 = encoder_l2(encoder_outputs1[0])
encoder_states2 = encoder_outputs2[1:]
#
decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])
#
decoder_l1 = tf.keras.layers.LSTM(200, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
decoder_l2 = tf.keras.layers.LSTM(200, return_sequences=True)(decoder_l1,initial_state = encoder_states2)
decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_output_features))(decoder_l2)
#
model_e2d2 = tf.keras.models.Model(encoder_inputs,decoder_outputs2)
#
model_e2d2.summary()
model_e2d2.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
history_e2d2=model_e2d2.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test),batch_size=32,verbose=1,callbacks=[lr_callback])


x_input = X_train[-1].reshape((1, n_past, n_features))
x_test_input = X_test[-1].reshape((1, n_past, n_features))
train_predict=model_e2d2.predict(x_input)
test_predict=model_e2d2.predict(x_test_input)
model_e2d2.save("src/model.keras")