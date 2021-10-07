import streamlit as st
import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
base = 'assets\\'

def app():

  st.title("나이브 베이즈를 이용한 스팸 분류")
  st.subheader("Supervised Learning")
  st.write('▼ emails.csv')
  spam_df =  pd.read_csv(base+'emails.csv')
  st.write(spam_df.head(3))
  st.write("- 5,574개의 이메일 메시지가 있으며, 스팸인지 아닌지의 정보를 가지고 있다.  \n"
           "- 컬럼 : text, spam  \n"
           "- spam 컬럼의 값이 1이면 스팸, 0이면 스팸이 아니다.  \n"
           "- 스팸인지 아닌지 분류하는 인공지능을 만든다  \n"
           "- 수퍼바이즈드 러닝의 분류 문제!")
  st.markdown('')

  st.markdown("## STEP #1: IMPORT DATASET")
  st.write("ex>")
  st.write(spam_df['text'][0])
  st.write(spam_df.describe())


  st.markdown("## STEP #2: VISUALIZE DATASET")

  st.write("스팸은 몇개이고, 아닌것은 몇개인지 확인")

  col, coll = st.columns(2)
  with col:
    st.write("- Spam")
    st.write(spam_df.loc[spam_df['spam'] == 1 , ].count()) # spam
  with coll:
    st.write("- Non Spam")
    st.write(spam_df.loc[spam_df['spam'] == 0 , ].count())

  #spam_df['spam'].value_counts()
  col1, coll1 = st.columns([6,4])
  with col1:
    fig1 = plt.figure()
    sns.countplot(data=spam_df, x ='spam')
    st.pyplot(fig1)


  st.markdown("## - 이메일의 길이가 스팸과 관련이 있는지 확인해보자.")
  st.markdown("### 이메일의 문자 길이를 구해서, length 라는 컬럼을 만든다.")

  st.write("spam_df['length'] = spam_df['text'].apply(len)")
  spam_df['length'] = spam_df['text'].apply(len)

  st.write("## 글자 길이를 히스토그램으로 나타낸다.")
  col2, coll2 = st.columns([6,4])
  with col2:
    fig2 = spam_df['length'].hist(bins=100)
    st.pyplot(fig2.get_figure())


  st.write("## 가장 긴 이메일을 찾아서 스팸인지 아닌지 확인하고, 이메일 내용을 확인한다.")

  st.write("spam_df.loc[spam_df['length'] == spam_df['length'].max(), ]")
  st.write(spam_df.loc[spam_df['length'] == spam_df['length'].max(), ])
  st.write(spam_df.loc[spam_df['length'] == spam_df['length'].max(),'text' ][2650][:100]+'...')

  st.write("## 파이차트를 통해, 스팸과 스팸이 아닌것이 몇 퍼센트인지 소수점 1자리 까지만 보인다")

  st.write(spam_df['spam'].value_counts())

  df_count = spam_df['spam'].value_counts()

  col3, coll3 = st.columns([6,4])
  with col3:
    fig2 = plt.figure()
    plt.pie(df_count, labels = ['0','1'] , autopct='%.1f')  #autopct 퍼센트
    plt.legend()
    st.pyplot(fig2)


  st.write("## 스팸이 아닌것은 ham 변수로, 스팸인것은 spam 변수로 저장한다.")
  ham = spam_df.loc[spam_df['spam']==0, ]

  spam = spam_df.loc[spam_df['spam']==1, ]

  st.write("## 스팸과 햄의 이메일 길이를 히스토그램으로 나타낸다.")
  col4, coll4 = st.columns(2)
  with col4:
    fig3 = plt.figure()
    plt.hist(spam['length'])
    st.pyplot(fig3)
  with coll4:
    fig4 = plt.figure()
    plt.hist(ham['length'])
    st.pyplot(fig4)

  st.subheader("==> 분석결과 길이와 스펨여부 관계는 의미가 없다. 문자열분석 해보자")

  col5, coll5 =st.columns([6,4])
  with col5:
    st.pyplot(spam['length'].hist().get_figure())

  st.markdown("## STEP #3: CREATE TESTING AND TRAINING DATASET/DATA CLEANING  \n"
              "### STEP 3.1 쉼표, 마침표 등의 구두점 제거하기  \n"
              "### STEP 3.2 STOPWORDS(불용어) 제거하기  \n"
              "### STEP 3.3 COUNT VECTORIZER ")

  st.markdown("")
  st.write("구두점 제거 함수 사용 (string.punctuation) ▶ ",string.punctuation)

  code3 = '''punc_removed =[ char for char in test: if char not in string.punctuation ]
             punc_removed_join = ''.join(punc_removed)
             punc_removed_join_clean =[ word for word in punc_removed_join.split() if word .lower() not in stopwords.words('english')]
             
             vectorizer = CountVectorizer()
             '''
  st.code(code3, language='python')

  st.markdown("## df의 이메일 내용을 Cleanning 한다.")
  st.write("- 1 구두점 제거  \n"
           "- 2 stopword 제거  \n"
           "- 3 count vec 적용")

  code = '''
  def message_cleaning (sentense) :
    punc_removed = [ char for char in sentense if char not in string.punctuation ]
    punc_removed_join = ''.join(punc_removed)
    punc_removed_join_clean = [ word for word in punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return punc_removed_join_clean  # 문자열 리스트 리턴
  '''

  st.code(code, language='python')
  def message_cleaning (sentense) :
    punc_removed = [ char for char in sentense if char not in string.punctuation ]
    punc_removed_join = ''.join(punc_removed)
    punc_removed_join_clean = [ word for word in punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return punc_removed_join_clean  # 문자열 리스트 리턴

  st.markdown("")
  st.write("▼ test")
  st.write("message_cleaning('hello world')")
  testing = message_cleaning ('hello world')
  st.code(testing, language='Python')

  #spam_df['text'].apply(message_cleaning)

  st.markdown("## 이제 이메일의 내용을 벡터라이징 한다.")
  code2 = '''
  vectorizer = CountVectorizer( analyzer= message_cleaning)  # 이 함수를 써서 data preprocessing 해라
  X = vectorizer.fit_transform( spam_df['text'])
  X.shape'''
  st.code(code2, language='python')

  vectorizer = CountVectorizer(analyzer=message_cleaning)  # 이 함수를 써서 data preprocessing 해라
  X = vectorizer.fit_transform(spam_df['text'])
  st.write(X.shape)


  st.markdown("## STEP #4: TRAINING THE MODEL WITH ALL DATASET")
  code4 ='''
  classifier = MultinomialNB() # model
  X # sparse matrix
  y = spam_df['spam']  # 정답
  '''
  st.code(code4, language='python')

  classifier = MultinomialNB()# 얘가 model
  y = spam_df['spam']  # 정답


  st.markdown("## STEP #4-1: Training셋과 Test셋으로 나눠서, 학습한다.")
  code5 = '''
  X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=7)
  classifier.fit(X_train, y_train)
  '''
  st.code(code5, language='python')
  X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=7)
  classifier.fit(X_train, y_train)

  st.markdown("## STEP #5: 테스트셋으로 평가한다.\n분류의 문제이므로 컨퓨전 매트릭스 확인.")
  code6 = '''
  y_pred = classifier.predict(X_test)
  cm = confusion_matrix(y_test, y_pred)
  cm
  '''
  st.code(code6, language='python')

  y_pred = classifier.predict(X_test)  # 얘가 api 서버에서 예측한다
  cm = confusion_matrix(y_test, y_pred)
  st.code(cm, language='python')
  st.write("## cm ▶ 실제값과 예측값을 비교하는 함수  \n"
           "실제는 0인데 1이라고 에측한것이 7  \n"
           "실제는 1인데 0이라고 예측한것이 4")
  st.code("(869 + 266) / cm.sum()", 'python')
  st.write((869 + 266) / cm.sum())  # 정확도 계산

  st.subheader("다음 2개의 문장을 테스트")
  st.write("1)")
  testing_sample = ['Free money!!!', "Hi Kim, Please let me know if you need any further information. Thanks"]
  st.code(testing_sample, 'python')

  code7='''
  classifier.predict(testing_sample) # error 
  \''' 모양이 안맞아 error 가 발생한다.
      학습시킬 모양 확인
      X_train.shape (4562, 37229) 2차원인데 testing_sample 은 1차원이다 \'''
  sample_X =vectorizer.transform(testing_sample) # 만들어놓은 vectorizer 이용
  sample_x.shape # shape 확인
  '''
  st.code(code7, 'python')

  sample_X = vectorizer.transform(testing_sample) # 이미만들어놓은 벡터라이저 이용해서 변환해라

  st.code(sample_X.shape, 'python')

  st.code("classifier.predict(sample_X)", 'python')
  st.code(classifier.predict(sample_X),'python')
  st.write("첫번째 - spam,  두번쨰 - ham")
  st.write("2)")
  testing_sample = ['Hello, I am Ryan, I would like to book a hotel in Bali by January 24th', 'money viagara!!!!!']
  st.code(testing_sample,'python')
  code8 = '''
  sample_2 = vectorizer.transform(testing_sample)
  classifier.predict(sample_2)
  '''
  st.code(code8,'python')
  sample_2 = vectorizer.transform(testing_sample)
  st.code(classifier.predict(sample_2), 'python')
  st.write("첫번째 - ham,  두번쨰 - spam")