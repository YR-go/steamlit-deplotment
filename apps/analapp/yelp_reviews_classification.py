import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix  # 분류문제만 metrics 쓰는거임
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud, STOPWORDS

base = 'assets\\'

def app():
  st.title("YELP 서비스의 리뷰 분석 (NLP)")
  st.header("STEP #1: IMPORT DATASET, 기본적인 통계 분석")
  code = '''
  yelp_df =  pd.read_csv('yelp.csv')
  yelp_df.loc[yelp_df['user_id']=='rLtl8ZkDX5vH5nAx9C3q5Q',].shape
    # id = 'rLtl8ZkDX5vH5nAx9C3q5Q' 인 유저가 남긴 리뷰를 확인
  '''
  st.code(code,'python')
  yelp_df =  pd.read_csv(base+'yelp.csv')
  st.code(yelp_df.loc[yelp_df['user_id']=='rLtl8ZkDX5vH5nAx9C3q5Q',].shape,'python')


  st.header("STEP #2: VISUALIZE DATASET")
  st.write("리뷰 길이와 별점의 관계를 분석한다")

  yelp_df['length'] = yelp_df['text'].apply(len)

  col1, coll1 = st.columns([4,6])
  with col1:
    st.write("리뷰 길이 시각화")
    fig1 = plt.figure()
    plt.hist(yelp_df['length'])
    st.pyplot(fig1)

  code1 ='''yelp_df[['stars','length']].corr()'''
  st.code(code1,'python')
  st.dataframe(yelp_df[['stars','length']].corr())
  st.write("▶ 관계없다")

  st.write("### 리뷰가 가장 긴 글을 찾아서, 리뷰 내용을 확인한다.")

  st.code("yelp_df.loc[yelp_df['length'] == yelp_df['length'].max() , ]",'python')
  st.dataframe(yelp_df.loc[yelp_df['length'] == yelp_df['length'].max() , ])
  st.code("yelp_df.loc[yelp_df['length'] == yelp_df['length'].max() , 'text'][55]",'python')
  st.code(yelp_df.loc[yelp_df['length'] == yelp_df['length'].max() , 'text'][55],'python')

  st.write("### 리뷰가 가장 짧은 리뷰는 총 몇개이며, 리뷰 내용은?")

  st.code(yelp_df.describe(), 'python')
  st.dataframe(yelp_df.loc[yelp_df['length'] == yelp_df['length'].min() , ])

  st.write("각 별점별로 리뷰가 몇개씩 있는지를 시각화 한다.")
  col3, coll3 = st.columns(2)
  with col3:
    fig2 = plt.figure()
    sns.countplot(data=yelp_df, x='stars').get_figure()
    st.pyplot(fig2)

  st.write("내림차순으로 정렬하여 시각화 한다.")
  col2, coll2 = st.columns(2)
  with col2:
    fig3 = plt.figure()
    sns.countplot(data=yelp_df, x='stars', order=yelp_df['stars'].value_counts().index)
    st.pyplot(fig3)

  st.subheader("긍정과 부정의 리뷰를 분석한다.")
  code2 = '''
  yelp_df_1 = yelp_df.loc[ yelp_df['stars']==1, ] # 별점 1점
  yelp_df_5 = yelp_df.loc[ yelp_df['stars']==5, ] # 별점 5점
  yelp_df_1_5 =pd.concat([yelp_df_1, yelp_df_5])  # 나중 train 을 위함
  yelp_df_1_5['stars'].unique()  
  '''
  yelp_df_1 = yelp_df.loc[ yelp_df['stars']==1, ]
  yelp_df_5 =yelp_df.loc[ yelp_df['stars']==5, ]
  yelp_df_1_5 =pd.concat([yelp_df_1, yelp_df_5])  # 나중 train 을 위함
  yelp_df_1_5['stars'].unique()

  col4, coll4 = st.columns(2)
  with col4:
    fig4 = plt.figure()
    sns.countplot(data=yelp_df_1_5, x='stars').get_figure()
    st.pyplot(fig4)

  st.write("이런 인발런스 한 데이터셋은 학습이 잘 안좋다.  \n"
           "균형적으로 맞춰서 (나노샘플링, fake data를 만들어서, 업샘플링) 데이터가 많을수록 학습이 잘 된다.")
  st.write("")
  st.write("### 별점 1점과 별점 5점의 리뷰의 비율이 나오도록, 파이차트로 시각화 한다.")

  col5, coll5 = st.columns(2)
  with col5:
    fig5 = plt.figure()
    plt.pie(yelp_df_1_5['stars'].value_counts(), autopct='%.1f%%', labels=['1★', '5★'])
    st.pyplot(fig5)


  st.subheader("앞의 과정들을 하나의 함수로 만든다.")
  st.write("1. 구두점 제거  \n"
           "2. 불용어 처리")

  code3 = '''
    stringPunctuation = \'!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~\'
    def message_cleaning (sentense) :
      punc_removed = [ char for char in sentense if char not in stringPunctuation ]
      punc_removed_join = ''.join(punc_removed)
      punc_removed_join_clean = [ word for word in punc_removed_join.split() if word.lower() not in stopwords.words('english')]
      return punc_removed_join_clean  # 문자열 리스트 리턴
    '''
  stringPunctuation ='''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)'''

  def message_cleaning(sentense):
    punc_removed = [char for char in sentense if char not in stringPunctuation]
    punc_removed_join = ''.join(punc_removed)
    punc_removed_join_clean = [word for word in punc_removed_join.split() if
                               word.lower() not in stopwords.words('english')]
    return punc_removed_join_clean  # 문자열 리스트 리턴


  st.code(code3, 'python')

  code4 = '''
    vectorizer = CountVectorizer(analyzer=message_cleaning)
    X = vectorizer.fit_transform(yelp_df_1_5['text'])
    X.shape'''

  vectorizer = CountVectorizer(analyzer=message_cleaning)
  X = vectorizer.fit_transform(yelp_df_1_5['text'])
  st.code(code4, 'python')
  st.code(X.shape,'python')

  st.header("STEP #4: 학습용과 테스트용으로 데이터프레임을 나눈다. 나이브베이즈 모델링 한다.")
  code5 = '''
  X
  y = yelp_df_1_5['stars']
  
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=34)
  
  classifier = MultinomialNB()
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)
  confusion_matrix(y_test, y_pred)'''

  y = yelp_df_1_5['stars']

  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=34)

  classifier = MultinomialNB()
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)
  cm = confusion_matrix(y_test, y_pred)

  st.code(code5,'python')
  st.write('▼ 정확도')
  st.code(((114+648) / cm.sum()), 'python')

  st.header("STEP #6 다음 문장이 긍정인지 부정인지 예측한다.")
  st.write("### 1. 'amazing food! highly recommmended'  \n"
           "### 2. 'shit food, made me sick'")

  sample = ['amazing food! highly recommmended', 'shit food, made me sick']
  sample_X = vectorizer.transform(sample)
  classifier.predict(sample_X)
  st.code(classifier.predict(sample_X),'python')

  st.title("WordCloud Visualizing")

  words = yelp_df_5['text'].tolist()
  words_as_one_string = ''.join(words)

  wc = WordCloud(background_color='white')
  fig6 = wc.generate(words_as_one_string)

  col6, coll6 = st.columns(2)
  with col6:
    plt.figure()
    plt.axis('off')  # 좌표없애기
    figg = np.array(fig6.to_image())
    st.image(figg)

  st.header("WordCloud + Stopwords = ?")
  wstopwords = STOPWORDS
  wstopwords.add('food')
  wc2 = WordCloud(background_color='pink', stopwords=wstopwords)
  wc_gen = wc2.generate(words_as_one_string)

  code6 = '''
  stopwords = STOPWORDS
  stopwords.add('food')
  wc2 = WordCloud(background_color='pink', stopwords=stopwords)
  wc_gen = wc2.generate(words_as_one_string)
  '''
  st.code(code6,'python')

  col7, coll7 = st.columns(2)
  with col7:
    plt.figure()
    plt.axis('off')
    fig7 = np.array(wc_gen.to_image())
    st.image(fig7)

  st.subheader("별점 1점 짜리 리뷰 시각화")

  bad_words = yelp_df_1['text'].to_list()
  bad_words_as_one_string = ''.join(bad_words)
  wc3 = WordCloud(background_color='black', stopwords=wstopwords)
  bad = wc3.generate(bad_words_as_one_string)

  col8 , coll8 = st.columns([4,6])
  with col8 :
    plt.figure()
    plt.axis('off')
    fig8 = np.array(bad.to_image())
    st.image(fig8)