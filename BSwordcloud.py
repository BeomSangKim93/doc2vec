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
#from multiprocessing import Pool
#import numpy as np

#파이썬 도표그리기 기본
#워드클라우드 그리기 위해
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

#영문리뷰자료 html을 문자열로 읽어와서 어간추출까지 마친리스트를 문자열로 변환하여 반환하는 함수
def review_to_words( raw_review ):
    # 1. HTML 제거
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. 영문자가 아닌 문자는 공백으로 변환
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. 소문자 변환
    words = letters_only.lower().split()
    # 4. 파이썬에서는 리스트보다 세트로 찾는게 훨씬 빠르다.
    # stopwords 를 세트로 변환한다.
    stops = set(stopwords.words('english'))
    # 5. Stopwords 불용어 제거
    meaningful_words = [w for w in words if not w in stops]
    # 6. 어간추출
    stemmer = SnowballStemmer('english')
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    # 7. 공백으로 구분된 문자열로 결합하여 결과를 반환
    return( ' '.join(stemming_words) )



# 참고 : https://gist.github.com/yong27/7869662
# http://www.racketracer.com/2016/07/06/pandas-in-parallel/


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


def displayWordCloud(data = None, backgroundcolor = 'white', width=800, height=600 ):
    wordcloud = WordCloud(stopwords = STOPWORDS, background_color = backgroundcolor, width = width, height = height).generate(data)
    plt.figure(figsize = (15 , 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()



# Import the pandas package, then use the "read_csv" function to read
# the labeled training data

train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
num_reviews = train['review'].size  #train 개수
print (train['review'][0][:100]) #잘읽어 왔는테 100개문자 출력

#2500개의 리뷰트레이닝 자료를 전부 전처리한다.
#전처리 결과를 clean_train_review 라는 리스트에 저장한다.
#5000개 단위로 상태를 찍도록 개선했다.
clean_train_reviews = []
for i in range(0, num_reviews):
     if (i + 1)%5000 == 0:
         print('Review {} of {} '.format(i+1, num_reviews))
     clean_train_reviews.append(review_to_words(train['review'][i]))
    

# 학습 데이터의 모든 단어에 대한 워드 클라우드를 그려본다.
displayWordCloud(' '.join(clean_train_reviews))