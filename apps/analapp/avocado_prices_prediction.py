from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
base = 'assets\\'
def app():
    st.title("AVOCADO 가격 예측 (Facebook Prophet )")
    st.header("STEP #1: 데이터 준비")
    avocado_df = pd.read_csv('avocado.csv')
    st.dataframe(avocado_df.head(3))

    st.header("STEP #2: EDA(Exploratory Data Analysis) :  탐색적 데이터 분석")
    st.write("필요없는 맨 처음 컬럼을 제거한다.")
    avocado_df.drop('Unnamed: 0', axis=1, inplace=True)
    st.code("avocado_df.drop('Unnamed: 0', axis=1, inplace=True)",'python')
    st.dataframe(avocado_df.head(3))
    st.write("뒤죽박죽인 데이터를 날짜로 정렬한다.")
    st.code('avocado_df.sort_values(by=\'Date\', ascending=\'True\', inplace=True)','python')
    avocado_df.sort_values(by='Date', ascending='True', inplace=True)
    st.write("날짜별로 가격이 어떻게 변하는지 간단하게 확인한다. Plot")
    meanData = avocado_df.groupby("Date")["AveragePrice"].mean()
    col1,coll1 =st.columns([6,4])
    with col1:
        fig1, ax1 = plt.subplots(figsize=(8,6))
        ax1.plot(meanData.index, meanData.values)
        plt.xticks(rotation = 45)
        st.pyplot(fig1)

    st.write("\'region\' 별로 데이터 몇개인지 시각화 한다.")

    st.code(avocado_df.groupby(by='region')['Date'].count(),'python')

    st.write("\'year\'별로 데이터가 몇건인지 확인한다.")

    st.code(avocado_df.groupby(by='year')['Date'].count(),'python')

    st.subheader("프로펫 분석을 위해, 두개의 컬럼만 가져온다. ('Date', 'AveragePrice')")
    avocado_prophet_df = avocado_df.loc[: , ['Date','AveragePrice']]
    st.write("avocado_prophet_df")
    st.dataframe(avocado_prophet_df.head(3))

    st.header("STEP #3: Prophet 을 이용한 예측 수행")
    st.write("ds 와 y 로 컬럼명을 셋팅한다.")
    st.code("avocado_prophet_df = avocado_prophet_df.rename(columns={'Date':'ds','AveragePrice':'y'})",'python')
    avocado_prophet_df = avocado_prophet_df.rename(columns={'Date':'ds','AveragePrice':'y'})

    st.subheader("Part 1: 프로펫 예측 한다.")
    code1 = '''
    # 365일치를 예측.
    m = Prophet()
    m.fit(avocado_prophet_df)
    
    future = m.make_future_dataframe(periods=365)
    price = m.predict(future)
    '''
    m = Prophet()
    m.fit(avocado_prophet_df)
    future = m.make_future_dataframe(periods=365)
    price = m.predict(future)

    st.code(code1, 'python')
    # 차트로 확인.
    col2,coll2 = st.columns([8,2])
    with col2:
        fig1 = m.plot(price)
        st.pyplot(fig1)

    col3,coll3 = st.columns([8,2])
    with col3:
        fig2 = m.plot_components(price)
        st.pyplot(fig2)

    st.subheader("Part 2: region 이 west 인 아보카도의 가격을 예측한다.")

    avocado_df_sample = avocado_df.loc[ avocado_df['region']=='West', ['Date','AveragePrice'] ]
    west_df = avocado_df_sample.rename(columns={'Date':'ds','AveragePrice':'y'})
    m2 = Prophet()
    m2.fit(west_df)
    west_price = m2.make_future_dataframe(365)
    future_west_price = m2.predict(west_price)

    col4, coll4 = st.columns([8,2])
    with col4:
        fig3 = m2.plot(future_west_price)
        st.pyplot(fig3)

    st.subheader("결론 : 전체적인 아보카도 가격은 하락하지만,  \n"
                 "웨스트 아보카도를 사면, 비싸게 팔수 있다.")