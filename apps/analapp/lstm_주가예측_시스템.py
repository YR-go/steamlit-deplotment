import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import FinanceDataReader as fdr
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

def app():
    st.title("LSTM 주가예측 시스템")
    code1 = '''
    start_date = '2011-09-01'
    
    samsung = fdr.DataReader('005930',start_date)
    samsung.head(3)
    '''
    start_date = '2011-09-01'
    samsung = fdr.DataReader('005930',start_date)
    st.dataframe(samsung.head(3))

    col1, coll1 = st.columns([8,2])
    with col1:
        fig1 = plt.figure(figsize=(16,9))
        sns.lineplot(y = samsung['Close'], x = samsung.index)
        plt.xlabel('Date')
        plt.ylabel('Stock')
        st.pyplot(fig1)

    st.header("Normalization")
    code2 = '''
    scalar = MinMaxScaler()
    scale_columns = ['Open','High','Low','Close','Volume']
    scaled = scalar.fit_transform(samsung[scale_columns])
    
    scaled'''

    scalar = MinMaxScaler()
    scale_columns = ['Open','High','Low','Close','Volume']
    scaled = scalar.fit_transform(samsung[scale_columns])

    st.code(code2,'python')
    st.code(scaled,'python')

    code3 = '''
    df = pd.DataFrame(data = scaled, columns=scale_columns)
    x_train, x_test, y_train, y_test = train_test_split(df.drop('Close',axis=1), df['Close'],
                                                        test_size=0.2, random_state=0,shuffle=False)
    
    x_train.shape, y_train.shape'''
    df = pd.DataFrame(data = scaled, columns=scale_columns)
    x_train, x_test, y_train, y_test = train_test_split(df.drop('Close',axis=1), df['Close'],
                                                        test_size=0.2, random_state=0,shuffle=False)

    st.code(code3,'python')
    st.code(x_train.shape, y_train.shape,'python')


    x_val = tf.reverse(x_train, [-1])
    y_val = tf.reverse(y_train, [-1])
    y_val = tf.expand_dims(y_val, axis=-1)

    x_train = tf.reverse(x_train, [-1])
    y_train = tf.reverse(y_train, [-1])
    #x_val.shape, y_val.shape

    st.header("Tensorflow 를 이용한 Sequence dataset 구성하기")
    code4 = '''
    def windowed_dataset(series, window_size, batch_size, shuffle):
      series = tf.expand_dims(series, axis=1)
    
      ds = tf.data.Dataset.from_tensor_slices(series)
      ds = ds.window(window_size +1, shift =1, drop_remainder=True)
      ds = ds.flat_map(lambda w: w.batch(window_size +1))
      if shuffle :
        ds = ds.shuffle(1000)
      ds = ds.map(lambda w: (w[ :-1], w[-1]))
      return ds.batch(batch_size).prefetch(1)
    
    WINDOW_SIZE = 20
    BATCH_SIZE= 32
    
    train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
    test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)
    
    for data in train_data.take(1):
        print(f'데이터셋(X) 구성(batch_size, window_size, feature갯수): {data[0].shape}')
        print(f'데이터셋(Y) 구성(batch_size, window_size, feature갯수): {data[1].shape}')
    
    '''
    st.code(code4,'python')
    def windowed_dataset(series, window_size, batch_size, shuffle):
      series = tf.expand_dims(series, axis=1)

      ds = tf.data.Dataset.from_tensor_slices(series)
      ds = ds.window(window_size +1, shift =1, drop_remainder=True)
      ds = ds.flat_map(lambda w: w.batch(window_size +1))
      if shuffle :
        ds = ds.shuffle(1000)
      ds = ds.map(lambda w: (w[ :-1], w[-1]))
      return ds.batch(batch_size).prefetch(1)

    WINDOW_SIZE = 20
    BATCH_SIZE= 32

    train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
    test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)

    for data in train_data.take(1):
        st.code(print(f'데이터셋(X) 구성(batch_size, window_size, feature갯수): {data[0].shape}'),'python')
        st.code(print(f'데이터셋(Y) 구성(batch_size, window_size, feature갯수): {data[1].shape}'),'python')

    st.header("Modeling")

    code5 = '''
    x_train = tf.expand_dims(x_train, axis=-1)
    y_train = tf.expand_dims(y_train, axis=1)
    x_train.shape, y_train.shape'''

    x_train = tf.expand_dims(x_train, axis=-1)
    y_train = tf.expand_dims(y_train, axis=1)
    st.code(x_train.shape, y_train.shape,'python')

    model = Sequential([Conv1D(filters=32, kernel_size=5,padding='causal',
                               activation='relu',input_shape=[WINDOW_SIZE, 1]),
                        LSTM(16, activation='tanh'),Dense(16, activation ='relu'),Dense(1)])

    model.compile(loss=Huber(),optimizer=Adam(0.0005), metrics='mse')

    history = model.fit(train_data, batch_size=BATCH_SIZE, epochs=50, verbose='auto')

    history = model.fit(train_data, batch_size=BATCH_SIZE, epochs=100, verbose=0)

    pred = model.predict(test_data)

    code6 = '''
    model = Sequential([Conv1D(filters=32, kernel_size=5,padding='causal',
                               activation='relu',input_shape=[WINDOW_SIZE, 1]),
                        LSTM(16, activation='tanh'),Dense(16, activation ='relu'),Dense(1)])
    
    model.compile(loss=Huber(),optimizer=Adam(0.0005), metrics='mse')
    
    history = model.fit(train_data, batch_size=BATCH_SIZE, epochs=50, verbose='auto')
    
    history = model.fit(train_data, batch_size=BATCH_SIZE, epochs=100, verbose=0)
    
    pred = model.predict(test_data)
    pred.shape'''

    st.code(pred.shape,'python')
    col2,coll2 = st.columns([8.2])
    with col2:
        fig1 = plt.figure(figsize=(12,9))
        plt.plot(np.asarray(y_test)[20:], label='actual')
        plt.plot(pred, label = 'prediction')
        plt.legend()
        st.pyplot(fig1)

    st.write("전날과 다음날과의 차이를 가지고 다시 만들어봐야겠다.")
