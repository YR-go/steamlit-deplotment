# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
base = 'assets\\'

def app():

    st.write("# 인구데이터 분석")
    st.write('https://mois.go.kr에서, 연령별 인구현황 통계표를 csv로 다운로드 한다. (남녀구문을 uncheck, 연령1세단위, 0~100, 전체읍면동현황)')

    df = pd.read_csv(base+'age.csv', encoding='cp949', thousands =',')
    st.write("'age.csv")
    st.write(df.head(3))

    st.write("수치 데이터 정보")
    st.write(df.describe())

    st.subheader("삼청동 지역정보 분석")

    dff = df.loc[ df['행정구역'].str.contains('삼청동'), ]
    st.write( dff)

    dff.loc[ 4 ,'2019년07월_계_0세': ]  # 연령별 인구수
    data =dff.loc[ 4 ,'2019년07월_계_0세': ].values  # 시각화

    col1, coll1 = st.columns([4,6])
    with col1:
        fig, ax = plt.subplots()
        ax.plot(data)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    st.subheader("실습 1. \'삼청동\' 의 인구 구조를,  0세부터 100세 까지 나이대 별로 몇명이 있는지 시각화 한다."
                 "- 가로축은 나이, 세로축은 인구수")

    st.write(df.loc[ df['행정구역'] == '서울특별시 종로구 삼청동(1111054000)' ,'2019년07월_계_0세': ])

    df_sam = df.loc[ df['행정구역'] == '서울특별시 종로구 삼청동(1111054000)' ,'2019년07월_계_0세': ].values

    df_sam = df_sam.transpose()

    col2, coll2 = st.columns([4,6])
    with col2:
        fig2, ax2 = plt.subplots()
        ax2.plot(df_sam)

        plt.xlabel('연령')
        plt.ylabel('인구수')
        plt.title('삼청동 연령별 인구수')

        st.pyplot(fig2)


    st.subheader("실습 2. \'종로구\' 의 인구 구조를,  0세부터 100세 까지 나이대 별로 몇명이 있는지 시각화 한다."
                 "- 가로축은 나이, 세로축은 인구수")

    new_df = df.loc[ df['행정구역'].str.contains('종로구'), ]
    data = new_df.iloc[0, 3:].values
    st.write(new_df.head(3))

    col3, coll3 = st.columns([4,6])
    with col3:
        fig3, ax3 = plt.subplots()
        ax3.plot(data)

        plt.xlabel('연령')
        plt.ylabel('인구수')
        plt.title('종로구 연령별 인구수')

        st.pyplot(fig3)



    st.subheader("실습 3. 위의 \'종로구\' 의 인구 구조를 만0세, 15, 25, 35, 45세 까지 5개 파이차트로, 각 인구수를 시각화 한다.")

    new_df = df.loc[ df['행정구역'].str.contains('종로구'), ]

    data = new_df.iloc[ 0, [3,3+10,3+20,3+30,3+40]].values # 구 데이터만0 numpy 로 가져옴

    my_label = ['만0세', '15', '25', '35', '45']
    st.write("my_label = ['만0세', '15', '25', '35', '45'] 로 설정")

    col4, coll4 = st.columns([4,6])
    with col4:
        fig4, ax4 = plt.subplots()
        ax4.pie(data, labels =my_label, autopct = '%d', wedgeprops={'width' :0.7})
        plt.title('종로구 인구구조 (만0세, 15, 25, 35, 45)')
        st.pyplot(fig4)

