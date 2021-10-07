# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib import font_manager, rc
base = "assets/"

def app():

    st.write(
        """# 지하철 유무임별 이용현황 데이터 분석
        """)

    st.subheader("자료출처 : t-money.co.kr 에서 자료를 제공함.")

    df = pd.read_csv(base+'subwayfee.csv', encoding='cp949')
    st.write('subwayfee.csv')
    st.write(df.head())

    st.write(df['사용월'].nunique(),'>> df[\'사용월\'].nunique() # 1 월의 데이터만 갖고있다.')
    st.write(df['지하철역'].nunique(),'>> df[\'지하철역\'].nunique()  # 총 509개의 역의 데이터이다.')

    st.write(df.describe())

    st.subheader("실습 0. 유임승차, 유임하차, 무임승차, 무임하차 4가지 별로, 각각 가장 많은 역을 조사한다.")

    st.write(df.loc[ df['유임승차'] == df['유임승차'].max() ,   ])

    st.write(df.loc[ df['무임승차'] == df['무임승차'].max() ,   ])

    st.write(df.loc[ df['무임승차'] == df['무임승차'].max() ,   ])

    st.write(df.loc[ df['무임하차'] == df['무임하차'].max() ,   ])

    st.subheader("실습 1. 무임승차 대비 유임승차 비율이 가장 높은 역?")

    df['비율'] = df['유임승차']/df['무임승차']
    st.write("새로운 colunm 생성"
             "df['비율'] = df['유임승차']/df['무임승차']")
    st.write(df.head(3))

    my_filter = df['비율'] == df['비율'].max()

    my_new = df.loc[ df['무임승차'] != 0 , ]

    st.write('my_new.loc[ my_new[\'비율\'] == my_new[\'비율\'].max(),]')
    st.write(my_new.loc[ my_new['비율'] == my_new['비율'].max(),])

    st.subheader("실습 2. 전체승차인원(유임+무임)이 만명이상인 역 중, 유임승차 비율이 가장 높은 역?")

    st.write(
        """####
        total = df.loc[ df['유임승차']+df['무임승차'] >= 10000 , ]
        
        total.loc[ total['유임승차']/total['무임승차'] == (total['유임승차']/total['무임승차']).max(),  ]
        """)

    total = df.loc[ df['유임승차']+df['무임승차'] >= 10000 , ]
    total.loc[ total['유임승차']/total['무임승차'] == (total['유임승차']/total['무임승차']).max(),  ]
    st.write(total.loc[ total['유임승차']/total['무임승차'] == (total['유임승차']/total['무임승차']).max(),  ])


    st.subheader("실습 3. 서울역의 유임승차, 유임하차, 무임 승차, 무임하차, 총 4개를 파이차트로 시각화한다.")

    seoul = df.iloc[0, 4:7+1]

    col1, coll1 = st.columns([6,4])
    with col1:
        fig, ax = plt.subplots()
        ax.pie(seoul, labels = seoul.index, autopct='%.1f')
        plt.title(df['지하철역'][0])

        st.pyplot(fig)





