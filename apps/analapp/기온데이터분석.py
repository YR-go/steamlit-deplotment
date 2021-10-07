# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import numpy as np
base = 'assets\\'

def app():
    st.write(
    """# 기온데이터 분석
    
    ### 1907년부터 2018년3월까지의 서울 기온 데이터를 CSV로 다운로드 한다.
    ### seoul.csv 파일을 열어 실행한다. encoding='cp949'
    """)

    temp = pd.read_csv(base+'seoul.csv', encoding='cp949')
    st.write(temp.head(3))

    st.subheader('날짜를 확인한다.')
    st.write(temp['날짜'].unique())



    st.subheader("실습 1. 가장 더운날은 언제인가?")

    st.write( temp.loc[ temp['최고기온(℃)'] == temp['최고기온(℃)'].max() , ])

    st.write(temp['날짜'])

    st.subheader("실습 2. 최고기온을 히스토그램으로 나타내기. (bin의 범위 = 4도)")

    st.write(temp.describe())

    xrange = np.arange(-17, 39+1,4)
    st.write("xrange = np.arange(-17, 39+1,4)로 설정한다")

    col , coll = st.columns([4,6])

    with col:
        fig1, ax1 = plt.subplots(figsize=(3, 5))
        ax1.hist(data=temp, x='최고기온(℃)', bins=xrange, rwidth=0.9)
        st.pyplot(fig1)

    with coll:
        st.empty()

    st.subheader("실습 3. 위는 모든 날짜에 대한 데이터다."
                "2014년도 부터의 데이터를 기준으로, bin의 범위를 4도로 만들어서, 히스토그램으로 보인다.")

    temp_after_2014 = temp.loc[ temp['날짜'] >= '2014' , ]
    st.write(temp_after_2014)
    x2range = np.arange(-11, 36+4,4) # range 다시 설정
    st.write('x2range = np.arange(-11, 36+4,4) # range 다시 설정')

    col2 , coll2 = st.columns([4,6])
    with col2:
        fig2, ax2 = plt.subplots(figsize=(3, 5))
        ax2.hist(data = temp_after_2014, x = '최고기온(℃)' , bins=x2range, rwidth=0.9)
        st.pyplot(fig2)
    with coll2:
        st.empty()

    st.subheader("실습 4.  2017-08-01 ~ 2017-08-15 사이의 날짜별(x축) 최고기온(y축)을 스케터 시각화한다.")

    temp_01_15 = temp.loc[ (temp['날짜']<= '2017-08-15') & (temp['날짜'] >= '2017-08-01')  , ]

    col3 , coll3 = st.columns([6,4])
    with col3:
        fig3, ax3 = plt.subplots()
        ax3.scatter(data = temp_01_15, x = '날짜', y='최고기온(℃)')
        plt.xticks(rotation=90)
        st.pyplot(fig3)

    with coll3:
        st.empty()


    st.write("plotly 차트를 많이 사용한다. 웹 대시보드 만들때 많이 사용한다.")



