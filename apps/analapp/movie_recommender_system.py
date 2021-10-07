import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
base = 'assets/'

def app():
  st.title("영화 추천 시스템")
  st.subheader("Item-based Collaborative Filtering 으로 추천시스템을 구현한다")
  st.write("사용자가 준 별점을 가지고 분석한다.  \n"
           "영화과 영화가 얼마나 유사도를 갖나 분석하기 위해 상관관계 분석을 한다.")
  col1, coll1 = st.columns([4,6])
  st.header("STEP #1: IMPORT DATASET")

  movie_titles_df = pd.read_csv(base+'Movie_Id_Titles.csv')
  with col1:
    st.write("[Movie_titiles]")
    st.dataframe(movie_titles_df.head(3))

  movies_rating_df =pd.read_csv(base + 'u.data', sep = '\t' , names = ['user_id', 'item_id', 'rating', 'timestamp'])  # sep = \t 탭으로 분리됨

  with coll1:
    st.write("[rating]")
    st.dataframe(movies_rating_df.head(3))

  code1 = '''
  # timestamp 컬럼은 필요없으니, movies_rating_df 에서 제거한다.
  movies_rating_df.drop('timestamp', axis=1, inplace=True)
  
  # 두개의 데이터프레임을 합친다.
  movies_rating_df = movies_rating_df.merge(movie_titles_df, on = 'item_id')
  movies_rating_df.head(3)
  '''
  st.code(code1,'python')

  # timestamp 컬럼은 필요없으니, movies_rating_df 에서 제거한다.
  movies_rating_df.drop('timestamp', axis=1, inplace=True)
  # 두개의 데이터프레임을 합친다.
  movies_rating_df = movies_rating_df.merge(movie_titles_df, on = 'item_id')
  st.dataframe(movies_rating_df.head(3))

  st.header("STEP #2: VISUALIZE DATASET")

  titles = movies_rating_df.groupby('title')
  code2 = '''
  # 각 영화 제목별로, 별점에 대한 기본분석을 진행한다.
  titles = movies_rating_df.groupby('title')
  titles.describe()'''
  st.code(code2, 'python')
  st.dataframe(titles.describe())

  code3 = '''
   # 각 영화별 별점의 평균을 구하고, 이를 ratings_df_mean 에 저장한다.
  ratings_df_mean = titles['rating'].mean()
  
  # 각 영화별로, 몇개의 데이터가 있는지 구하고, 이를 ratings_df_count 에 저장한다.
  ratings_df_count = movies_rating_df.groupby(by = 'title')['rating'].count()
  '''
  st.code(code3,'python')

  # 각 영화별 별점의 평균을 구하고, 이를 ratings_df_mean 에 저장한다.
  ratings_df_mean = titles['rating'].mean()
  # 각 영화별로, 몇개의 데이터가 있는지 구하고, 이를 ratings_df_count 에 저장한다.
  ratings_df_count = movies_rating_df.groupby(by = 'title')['rating'].count()

  st.write("이제 신뢰도 파악이 가능해졌다")

  st.write("### 두 데이터프레임을 합친다.  \n"
           "#### - join은 2차원 이상일 때만 사용 / - concat 은 컬럼이 똑같을 때 사용")

  code3 = '''
  ratings_mean_count_df =  pd.concat([ratings_df_count,ratings_df_mean], axis=1)
  ratings_mean_count_df.head(3) # 오류발생 !
  '''
  st.code(code3, 'python')
  st.code("st.write(), st.dataframe() 에서 모두 value error가 발생한다.")
  st.image(base+"valueerror.JPG",width=500, use_column_width=[1,2,1])

  ratings_mean_count_df =  pd.concat([ratings_df_count,ratings_df_mean], axis=1)

  code4 ='''
  # 컬럼명을 확인하면, 합쳐진 컬럼들이 rating, rating 이라고 되어있다.
  # 이를 count, mean 으로 컬럼명을 셋팅한다.
  ratings_mean_count_df.columns=['count' ,'mean']
  '''
  st.code(code4,'python')
  ratings_mean_count_df.columns=['count' ,'mean']

  col1, colll1, coll1 = st.columns([2,0.5,2.5])
  with col1:
    st.write("mean 으로 히스토그램을 그려본다.")
    fig1 = plt.figure()
    ratings_mean_count_df['mean'].hist()
    st.pyplot(fig1)
  with coll1:
    st.write("count 로 히스토그램을 그려본다.")
    fig2 = plt.figure()
    ratings_mean_count_df['count'].plot(kind='hist')
    st.pyplot(fig2)

  st.write("평균점수가 5점인 영화들은 어떤 영화인지 확인한다.")
  st.dataframe(ratings_mean_count_df.loc[ ratings_mean_count_df['mean']==5,])

  st.write("count 가 가장 많은 것부터 정렬하여 100개까지만 보인다.")
  st.dataframe(ratings_mean_count_df.sort_values(by = 'count', ascending=False).head(100))

  st.header("STEP #3: 영화 하나에 대한, ITEM-BASED COLLABORATIVE FILTERING 수행")
  st.write("- movies_rating_df 를 Pivot Table 하여 콜라보레이티브 필터링 포맷으로 변경한다.")
  st.dataframe(movies_rating_df.pivot_table(index='user_id', columns='title',values='rating').head(3))

  userid_movietitle_metrix = movies_rating_df.pivot_table(index='user_id', columns='title',values='rating')

  col2 , coll2 = st.columns([7,3])
  with col2:
    st.write("userid가 23인 사람이 'Star Wars (1977)을 보고 매긴 점수는?")

  with coll2:
    st.write(userid_movietitle_metrix.loc[ 23, 'Star Wars (1977)'])

  st.write('')
  st.subheader("이제")
  st.markdown("- 전체 영화와 '타이타닉' 영화의 상관관계 분석을 한 후 \n"
           "- 타이타닉을 본 사람들에게 상관계수가 높은 영화를 추천하면 된다.  \n"
           "- corrwith 함수를 이용한다.")

  st.write("\'Titanic (1997)\' 과 다른 영화들 간의 상관관계 분석-corrwith()  \n"
           "추천시스템의 정확도를 위해서는 영화를 본 사람이 적어도 80명 이상인 데이터를 사용한다.")

  titanic_series =userid_movietitle_metrix.corrwith ( userid_movietitle_metrix['Titanic (1997)'] )

  st.subheader("즉 타이타닉을 봤을 때 추천할 수 있는 영화는")

  titanic_correlations = pd.DataFrame(titanic_series, columns=['Correlation'])

  titanic_correlations = titanic_correlations.join( ratings_mean_count_df['count'])

  titanic_correlations.dropna(inplace=True)

  titanic_correlations_over80 = titanic_correlations.loc[ titanic_correlations['count']>= 80, ]

  st.dataframe(titanic_correlations_over80.sort_values(by= 'Correlation',ascending=False).head(7))
  col3,coll3 = st.columns([8.5,1.5])
  with coll3:
    st.write("...등이다.")

  st.header("STEP #4: 'star wars' 를 본 사람들에게 영화를 추천한다. 5개의 추천 영화 제목을 찾는다.")
  st.write("먼저 star wars 의 정확한 이름을 검색해서 찾는다.  \n"
           "그리고 나서 스타워즈를 본 유저의 데이터를 가져와서, 위와 같이 상관관계분석을 한다.")
  code5 = '''
  movie_titles_df.loc[movie_titles_df['title'].str.lower().str.contains('star'),]
  '''
  st.code(code5,'python')
  st.dataframe(movie_titles_df.loc[movie_titles_df['title'].str.lower().str.contains('star'), ])

  code6 = '''
  starwars_data = userid_movietitle_metrix.corrwith(userid_movietitle_metrix['Star Wars (1977)'])
  starwars = pd.DataFrame(starwars_data, columns=['Correlation'])
  starwars = starwars.join(ratings_mean_count_df['count'])
  starwars = starwars.loc[starwars['count'] >=80 ,]
  starwars.dropna(inplace=True)
  starwars.sort_values(by='Correlation', ascending=False).head(5)
  '''
  st.code(code6, 'python')
  starwars_data = userid_movietitle_metrix.corrwith(userid_movietitle_metrix['Star Wars (1977)'])

  starwars = pd.DataFrame(starwars_data, columns=['Correlation'])

  starwars = starwars.join(ratings_mean_count_df['count'])

  starwars = starwars.loc[starwars['count'] >=80 ,]

  starwars.dropna(inplace=True)

  st.dataframe(starwars.sort_values(by='Correlation', ascending=False).head(5))

  st.header("STEP #5: 전체 데이터셋에 대한 ITEM-BASED COLLABORATIVE FILTER 를 만든다.")
  st.write("적어도 80명 이상이 점수를 준 영화를 대상으로 상관계수를 뽑는다.")

  st.code("userid_movietitle_metrix.corr(min_periods=80)  # uid 가 80개 이상인 것만 측정",'python')

  movie_correlations = userid_movietitle_metrix.corr(min_periods=80)

  st.subheader("나의 별점 정보를 가지고, 영화를 추천 받아본다.")

  myRatings = pd.read_csv(base +'My_Ratings.csv')
  st.write("My_Ratings.csv")
  st.dataframe(myRatings)

  code7 = '''
  movie_title = myRatings['Movie Name'][0]
  reco_movie1 = movie_correlations[movie_title].dropna().sort_values(ascending = False).to_frame()
  reco_movie1.rename(columns={ 'movie_title' : 'Correlation'}, inplace=True)
  reco_movie1['Weight'] = reco_movie1['Correlation'] * myRatings['Ratings'][0]   # 내 점수라는 가중치 반영됨
  '''
  movie_title = myRatings['Movie Name'][0]

  reco_movie1 = movie_correlations[movie_title].dropna().sort_values(ascending = False).to_frame()

  reco_movie1.rename(columns={ movie_title : 'Correlation'}, inplace=True)

  reco_movie1['Weight'] = reco_movie1['Correlation'] * myRatings['Ratings'][0]   # 내 점수라는 가중치 반영됨

  st.header("STEP #6: 위의 추천영화 작업을 자동화 하기 위한 파이프라인을 만든다.")
  st.write("반복문을 사용하여 비슷한영화에 대한 데이터프레임을 만들고,  \n"
           "이를 아래 빈 데이터프레임에 계속하여 추가한다.  \n"
           "반복문이 끝나면, 아래 데이터프레임을 wegiht 컬럼으로 정렬한다.")

  code8 = '''
  similar_movies_list = pd.DataFrame()

  for i in range(0, len(myRatings)) :
    movie_title = myRatings['Movie Name'][i]
    similar_movie = movie_correlations[movie_title2].dropna().sort_values(ascending=False).to_frame()
    similar_movie.rename(columns={movie_title2:'Correlation'}, inplace=True)
    similar_movie['Weight'] = similar_movie['Correlation'] * myRatings['Ratings'][i]
    similar_movies_list = similar_movies_list.append(similar_movie)

  similar_movies_list.sort_values(by='Weight', ascending=False)
  '''
  st.code(code8,'python')
  similar_movies_list = pd.DataFrame()

  for i in range(0, len(myRatings)) :
    movie_title = myRatings['Movie Name'][i]
    similar_movie = movie_correlations[movie_title].dropna().sort_values(ascending=False).to_frame()
    similar_movie.rename(columns={movie_title:'Correlation'}, inplace=True)
    similar_movie['Weight'] = similar_movie['Correlation'] * myRatings['Ratings'][i]
    similar_movies_list = similar_movies_list.append(similar_movie)

  st.write("내게 추천 해 줄 영화는 다음과 같다.")
  st.dataframe(similar_movies_list.sort_values(by='Weight', ascending=False).head(5))
