import streamlit as st
import pandas as pd
import numpy as np
from data.create_data import create_table

def app():
    st.title('AI Parking Lot System')
    st.subheader('team project')

    st.image('assets\\res.png', width=240, caption='Parking Lot\'s status in App')


    st.markdown('')
    st.markdown('## ◆ part : ML, Image processing')
    st.markdown('')

    st.subheader('1. [Image processing]')
    st.write('OpenCV를 활용, 주차구역을 이미지 좌표를 받아 Crop 한다')
    st.markdown('### о 판별할 주차장 영상')
    st.image('assets\image05.png')
    st.markdown('')
    st.markdown('### о 좌표값 설정')
    st.image('assets\image04.png')

    st.markdown('')

    st.subheader('2. [ML]')
    st.write('Tensorflow, Keras, MobileNet V2, Global Average Pooling 활용 ')
    st.write('(sorce code)')
    st.write('https://github.com/YR-go/Parking-Lot-Project/tree/main/%5B%EC%98%81%EC%83%81%20%EC%A0%95%EB%B3%B4%20%EB%B6%84%EC%84%9D%20-%20Model%20%ED%95%99%EC%8A%B5%5D')
    st.markdown('### о 판별된 주차장 구역')
    col1, col2 = st.columns(2)
    with col1:
        st.image('assets\캡처3.JPG', caption='주차된 경우')
    with col2:
        st.image('assets\캡처.JPG', caption='비어있는 경우')

    st.markdown('')
    st.markdown('### о 예측결과 출력')
    st.write('1 번쨰칸) ■ 2 번쨰칸) ■ 3 번쨰칸) ■ 4 번쨰칸) ■ 5 번쨰칸) □ 6 번쨰칸) ■ 7 번쨰칸) ■ 8 번쨰칸) ■ 9 번쨰칸) ■ 10 번쨰칸) ■ 11 번쨰칸) ■ 12 번쨰칸) ■ 13 번쨰칸) ■ 14 번쨰칸) ■ 15 번쨰칸) ■ 16 번쨰칸) ■ 17 번쨰칸) ■ 18 번쨰칸) ■ 19 번쨰칸) ■ 20 번쨰칸) ■ 21 번쨰칸) ■ 22 번쨰칸) ■ 23 번쨰칸) ■ 24 번쨰칸) ■ 25 번쨰칸) ■ 26 번쨰칸) □ 27 번쨰칸) ■ 28 번쨰칸) □')