# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd # Import Pandas for data manipulation using dataframes
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sb # Statistical data visualization
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
base = 'assets\\'

def app():
  st.write("# Amazon 리뷰 분석")
  st.write("목적 : 리뷰로부터 인사이트를 얻고, 감성분석을 한다"
           "Wordcloud 를 사용하여 리뷰데이터를 분석한다"
           "- Dataset: www.kaggle.com/sid321axn/amazon-alexa-reviews")

  df = pd.read_csv(base+'amazon_alexa.tsv', sep='\t')
  st.write('amazon_alexa.tsv')
  st.write(df.head())

  st.subheader("실습 1. rating 과 varified_reviews 분석"
               "feedback 1은 긍정 0은 부정")
  st.markdown('')
  st.write("verified_reviews 컬럼의 내용 확인")
  st.write(df['verified_reviews'])
  st.write(df.describe())



  st.subheader("실습 2. 긍정 리뷰와 부정리뷰의 갯수를 그래프로 시각화")

  col1, coll1 =st.columns([6,4])
  with col1:
    fig1 = sb.countplot(data =df, x = 'feedback')
    st.pyplot(fig1.get_figure())

  st.write("긍정 리뷰 수, 부정 리뷰 수")
  st.write(df['feedback'].value_counts())

  st.subheader("실습 3. 유저의 별점(rationg) 별 리뷰갯수를 그래프로 시각화")

  index_order = df['rating'].value_counts().index

  col2, coll2 = st.columns([6,4])
  with col2:
    fig2 = sb.countplot(data =df, x = 'rating',order=index_order) # 정렬 후 차트
    st.write("sb.countplot(data =df, x = 'rating',order=index_order) # 정렬 후 차트")
    st.pyplot(fig2.get_figure())

  st.markdown("## WORD CLOUD 사용하여, 유저들이 어떤 단어를 많이 사용하였는지 시각화 한다.")
  st.markdown('')
  st.subheader("실습 4. verified_reviews 를 하나의 리스트로 만든다.")

  review_list = df['verified_reviews'].tolist()  # 하나의 리스트로 만들고나면 리스트 조인이 가능. 거대한 문자열로 만들어서 워드클라우드 사용해야함

  st.write(review_list[:5])
  st.markdown('### ...')
  st.subheader(" 실습 5. 위의 words 리스트를 " " 으로 합쳐서 하나의 문자열로 만든다.")
  st.markdown('')
  review_as_one_string = " ".join(review_list)  # ' ' 공백으로 합치기
  st.write(review_as_one_string[:200]+'...')


  st.subheader(" 실습 6. WordCloud 를 이용하여 많이 나온 단어들을 시각화 한다.")

  wc = WordCloud(background_color='orange', max_words=500) # 글자갯수 500개로 셑
  wc_gen = wc.generate(review_as_one_string)

  col3, coll3 = st.columns([6,4])
  with col3:
    plt.figure(figsize=(10,13))
    plt.axis('off') # 좌표없애기
    fig3=np.array(wc_gen.to_image())
    st.image(fig3)


  st.subheader("실습 7. Data Cleaning / Feature Engineering")
  st.write("- sklearn 이용")
  st.write("- 분석을 위해 문자를 숫자로 바꾼다.")
  st.write("- 단어를 알파벳 순으로 갯수 count")


  vectorizer = CountVectorizer()

  st.write("vectorizer.fit_transform(df['verified_reviews'])")
  st.write(vectorizer.fit_transform(df['verified_reviews']))
  st.write("""#### fit 학습하라(단어 뽑아서 정렬해서 컬럼으로 만들어라)""")
  st.write("""#### transform 학습하고 바궈서 간져와라 (숫자 집어넣어서 가져와라)""")
  st.write("""#### (3150 4044) 리뷰갯수 3150개, 컬럼수가 4044개 리뷰제거 후 단어갯수가 4044개""")
  st.markdown("")
  alexa_vc = vectorizer.fit_transform(df['verified_reviews'])

  vectorizer.get_feature_names() # 백터라이져가 이 단어를 숫자로 바꿔준거다
  st.write("vectorizer.get_feature_names() ◀ 백터라이져가 이 단어를 숫자로 바꿔준다")
  st.write(vectorizer.get_feature_names()[:3])
  st.write("....")
  len(vectorizer.get_feature_names())

  alexa_vc.toarray()

  alexa_vc.shape

  st.write("첫 번째 review = " + df['verified_reviews'][0])

  word_count_array = alexa_vc.toarray()

  col4, coll4 = st.columns([6,4])
  with col4:
    fig4, ax4 = plt.subplots()
    ax4.plot(word_count_array[ 0, : ])  # numpy[행,렬] , 시각화
    st.pyplot(fig4)


  st.write("두 번째 review = "+df['verified_reviews'][1])

  col5, coll5 = st.columns([6,4])
  with col5:
    fig5, ax5 = plt.subplots()
    ax5.plot(word_count_array[ 1, : ])  # numpy[행,렬] , 시각화
    st.pyplot(fig5)

  st.write("세 번째 review = "+df['verified_reviews'][2])

  col6, coll6 = st.columns([6,4])
  with col6:
    fig6, ax6 = plt.subplots()
    ax6.plot(word_count_array[ 2, : ])  # numpy[행,렬] , 시각화
    st.pyplot(fig6)


  st.title( "리뷰의 글자갯수와 별점은 관계가 있을까?")
  st.header("- 리뷰를 길게 쓰면 별점을 잘 줄까?")
  st.header("- 리뷰를 짧게 쓰면 별점을 잘 줄까?")
  st.header(" 숫자로 바꾼 review들로 분석을 해본다")

  count = []

  for i in range(0, 3150):
    count.append(word_count_array[i, :].sum())

  count

  word_count_array.shape

  st.write("review 단어 수를 count 한다")
  df['count'] = count

  df.head()

  """- 관계없다."""
  st.write("count 와 rating을 비교해본다.")
  st.write("df[ ['rating', 'count']].corr()")
  col11, coll11 = st.columns([4,6])
  with col11:
    st.write(df[ ['rating', 'count']].corr())
  with coll11:
    st.write(" ▶ 상관계수가 0에 가깝다. 관계가 없다.") # 관계없다

  st.subheader("리뷰길이와 별점의 관계 시각화")
  col7,coll7 = st.columns(2)
  with col7:
    fig6 =  sb.lmplot(x="rating", y="count", data=df)
    st.pyplot(fig6)
  with coll7:
    fig7 = sb.pairplot(data = df, vars=['rating', 'count'])
    st.pyplot(fig7)

  df['review len'] = df['verified_reviews'].apply(len)

  df['verified_reviews'].sum()

  st.subheader("두 컬럼간의 관계를 시각화 할 때는 Scatter 를 사용한다.")
  st.write("- scatter pairplot regplot 사용하자")

  col9, coll9 = st.columns([6,4])
  with col9:
    fig8, ax8 = plt.subplots()
    ax8.scatter(data = df, x='rating', y= 'count')
    st.pyplot(fig8)

  st.write("- data들이 너무 밀접해있어 알아볼 수가 없다.  ▶ heatmap을 사용한다.")

  col10, coll10 = st.columns([6,4])
  with col10:
    fig9 = plt.figure()
    plt.hist2d(data=df, x='rating', y='count', cmin=0.5, bins=20)
    plt.colorbar()
    st.pyplot(fig9)