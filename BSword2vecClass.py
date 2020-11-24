#파이썬기본 panddas bs4, re
import pandas as pd       
from bs4 import BeautifulSoup
import re


# Import the stop word list
# nltk에서 제공하는 영문 stopwords
from nltk.corpus import stopwords 
#토큰한 단어에서 어간 추출하는 부분 stemming
from nltk.stem.snowball import SnowballStemmer
#Lemmatization 음소표기법 동음 이의어의 내용 그룹화함
from nltk.stem import WordNetLemmatizer


#멀티프로세스로 실행
from multiprocessing import Pool
import numpy as np

#파이썬 도표그리기 기본
#워드클라우드 그리기 위해
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt



# 참고 : https://gist.github.com/yong27/7869662
# http://www.racketracer.com/2016/07/06/pandas-in-parallel/


class BSword2vecUtility(object):

    @staticmethod
    #영문리뷰자료 html을 문자열로 읽어와서 어간추출까지 마친리스트를 문자열로 변환하여 반환하는 함수
    def review_to_wordslist( raw_review, remove_stopwords=False ):
        # 1. HTML 제거
        review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
        # 2. 영문자가 아닌 문자는 공백으로 변환
        letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
        # 3. 소문자 변환
        words = letters_only.lower().split()
        # 4. Stopwords 불용어 제거
        if remove_stopwords:
            # stopwords 를 세트로 변환한다.
            stops = set(stopwords.words('english'))
            # Stopwords 불용어 제거
            words = [w for w in words if not w in stops]
        # 5. 어간추출
        stemmer = SnowballStemmer('english')
        words = [stemmer.stem(w) for w in words]
        # 6. 리스트를 반환
        return(words)


    @staticmethod    
    #영문리뷰자료 html을 문자열로 읽어와서 어간추출까지 마친리스트를 문자열로 변환하여 반환하는 함수
    def review_to_words( raw_review, remove_stopwords=False ):
        # 1. HTML 제거
        review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
        # 2. 영문자가 아닌 문자는 공백으로 변환
        letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
        # 3. 소문자 변환 리스트로 변환
        words = letters_only.lower().split()
        # 4. Stopwords 불용어 제거
        if remove_stopwords:
            # stopwords 를 세트로 변환한다.
            stops = set(stopwords.words('english'))
            # Stopwords 불용어 제거
            words = [w for w in words if not w in stops]
        # 6. 어간추출
        stemmer = SnowballStemmer('english')
        words = [stemmer.stem(w) for w in words]
        # 7. 공백으로 구분된 문자열로 결합하여 결과를 반환
        return( ' '.join(stemming_words) )



#멀티프로세스를 사용해서 성능을 빠르게 하기 위해 사용
    def _apply_df(args):
        df, func, kwargs = args
        return df.apply(func, **kwargs)

    def apply_by_multiprocessing(df, func, **kwargs):
        # 키워드 항목 중 workers 파라메터를 꺼냄
        workers = kwargs.pop('workers')
        # 위에서 가져온 workers 수로 프로세스 풀을 정의
        pool = Pool(processes=workers)
        # 실행할 함수와 데이터프레임을 워커의 수 만큼 나눠 작업
        result = pool.map(_apply_df, [(d, func, kwargs)
                for d in np.array_split(df, workers)])
        pool.close()
        # 작업 결과를 합쳐서 반환
        return pd.concat(list(result))

#워드 클라우드 그리기 
    def displayWordCloud(data = None, backgroundcolor = 'white', width=800, height=600 ):
        wordcloud = WordCloud(stopwords = STOPWORDS, background_color = backgroundcolor, width = width, height = height).generate(data)
        plt.figure(figsize = (15 , 10))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()



