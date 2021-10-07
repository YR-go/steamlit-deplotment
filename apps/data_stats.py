import streamlit as st
from apps.analapp import 기온데이터분석,대중교통,인구조사,amazon_reviews_analysis,movie_recommender_system

def app():
    cola, colb = st.columns([2,1])
    with cola:
        st.title('Data Analysis')
    with colb:
        end = st.button('프로젝트 접기')


    p1 = st.button('1. 프로젝트_기온데이터분석')
    if p1 :
        기온데이터분석.app()
    p2 = st.button("2. 대중교통 이용정보분석")
    if p2:
        대중교통.app()
    p3 = st.button('3. 인구데이터 분석')
    if p3:
        인구조사.app()
    p4 = st.button('4. Amazon 리뷰 분석')
    if p4:
        amazon_reviews_analysis.app()
    p5 = st.button("5. 영화 추천 시스템")
    if p5 :
        movie_recommender_system.app()





    colc, cold = st.columns([2, 1])
    with cold:
        end2 = st.button('접기')

    if end2:
        st.caching.clear_cache()

'''
    st.write("This is a sample data stats in the mutliapp.")
    st.write("See `apps/data_stats.py` to know how to use it.")

    st.markdown("### Plot Data")
    df = create_table()

    st.line_chart(df)

 '''