import streamlit as st


def app():
    st.title('Cafe App with NFC')
    st.subheader('team project')

    col1, col2, col3 = st.columns([4,1,9])

    with col1:
        st.subheader('<APP>')
        #st.markdown('<img src=\'https://user-images.githubusercontent.com/56214404/134175859-8edee860-3e0c-4099-a3a8-dd73d4ed2ae1.gif\' width=\'200\' height=\'500\' />')
        st.image("https://user-images.githubusercontent.com/56214404/134175859-8edee860-3e0c-4099-a3a8-dd73d4ed2ae1.gif", width=200)

    with col2:
        st.empty()

    with col3:
        st.subheader('<POS>')
        st.image("https://user-images.githubusercontent.com/56214404/134176617-fce3436f-6ca2-44d1-a2d7-20ed30ddf4ed.gif", width=500)

    st.markdown("")
    st.markdown('## ◆ Part : Web, DB')
    st.markdown("")
    st.subheader("1. [Web] ▶ https://cafe-f6a97.web.app/")
    st.image('assets/pos.JPG', width=500, caption='Web Page')
    st.markdown("")

    st.subheader('2. [DB] (Google Firebase)')
    st.image('assets/dbsize.jpg', width=500)
    st.image('assets/posrecordsize.png', width=400)
