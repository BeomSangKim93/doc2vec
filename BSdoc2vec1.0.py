# 출력이 너무 길어지지 않게하기 위해 찍지 않도록 했으나 
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import csv
from tqdm import tqdm


#형태소 분석기 로딩
from konlpy.tag import Kkma
from konlpy.tag import Twitter
from konlpy.tag import Okt
from nltk import word_tokenize

# gensim의 doc2vec 이용
#from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#from nltk.tokenize import word_tokenize
from gensim.models import doc2vec
import multiprocessing
from gensim.models.doc2vec import TaggedDocument
import gensim

#글로벌 변수
global num_train, DATA_PATH, MODEL_PATH, load_train_file, load_test_file, save_clean_file, stopwords
global modelfile, word2vec_file, LabeledSentence

LabeledSentence = gensim.models.doc2vec.LabeledSentence


#파일이름 세팅하기
DATA_PATH = 'data/'
MODEL_PATH = 'models/'
load_train_file = DATA_PATH +'ratings_train.txt'
load_test_file = DATA_PATH +'ratings_test.txt'
save_clean_file = DATA_PATH +'clean_naver_okt.csv'
modelfile = MODEL_PATH + 'doc2vec.model'
word2vec_file = modelfile + '.word2vec_format'

num_train = 15

def make_CleanData():

    global num_train, DATA_PATH, MODEL_PATH, load_train_file, load_test_file, save_clean_file, stopwords

    #네이버 영화 리뷰 데이터를 읽어 온다.
    kr_train = pd.read_csv(load_train_file, header=0, delimiter='\t', quoting=3)

    #네이버 영화 읽오왔는지 확인
    print(kr_train['document'].size)

    #1. 중복 데이터 제거
    kr_train.drop_duplicates(subset = ['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거

    #2.정규 표현식을 통한 한글 외 문자 제거
    kr_train['document'] = kr_train['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")# 정규 표현식 수행
    
    #3.빈내용은 Null로 표기
    kr_train['document'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경

    #4. Null 값 제거
    kr_train = kr_train.dropna(how = 'any') # Null 값이 존재하는 행 제거
    print(kr_train.isnull().values.any()) # Null 값이 존재하는지 확인

    #4.1 트레이닝 개수 셋팅
    num_train =  kr_train['document'].size

    #5. 불용어 셋팅
    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    stopwords = set(stopwords) #속도 개선을 위해 set 집합으로 변환
    print('불용어: {}'.format(stopwords))

    #5. 형태소 분석, DOC2VEC를 학습시키기 위해서는 Tag값이 필요하다
    #6. 불용어 제거
    #$$$ 시간이 걸림
    # 형태소 분석기 OKT를 이용해서 형태로 분석
    # 한글 형태소 분석기로 분석하고, 불용어 제거하고 2단계로 영어보다 단순함.
    i=0
    okt = Okt()
    tokenized_data = []
    df = pd.DataFrame(kr_train)
    kr_train_list = df.values.tolist() #판다스 리스트로 변환
    print('파일 읽어옴')
#   return
    
    for row in kr_train_list:  #padaa에 index가 잘못되어 있어서 잘 실행됨
        temp_X = okt.morphs(row[1], stem=True) # 토큰화
        temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
        tags = row[2]
        temp_X = [temp_X, str(tags)]
        tokenized_data.append(temp_X)
        i += 1
        if i%10000 == 0:
            print('문서 전처리 {} of {} '.format(i, num_train))
#            break #여기까지만 데이터 사용
            
#    print(tokenized_data)

    #7. 크린 데이터 파일로 저장, 한줄씩 문자열로
    #$$$$ 문자열로 변환하여 한줄씩 저장
    #토큰 문자열로 변환하고 한줄씩 추가
    clean_train_str = ''
    for token in tokenized_data:
        clean_train_str += token[1]+','
        clean_train_str += ' '.join(token[0])
        clean_train_str += '\n'

    #clean_train_reviews  파일로 저장
    with open(save_clean_file, "w", encoding="utf-8") as fp:
        fp.write(clean_train_str)





#################################################################
### 크린데이터 트레이닝 하기
################################################################

class LabeledLineSentence(object):
    global LabeledSentence
    def __init__(self, docs, tags):
        self.labels_list = tags
        self.doc_list = docs
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(words=doc,tags=[self.labels_list[idx]])

###########################################################
def BSdoc2vectrain():
    global modelfile, word2vec_file
    cores = multiprocessing.cpu_count()

    # 8. Gensim로딩하고 doc2vec 초기셋팅하기
    #doc2vec parameters
    vector_size = 300   #벡터 사이즈
    window_size = 15    #윈도우 사이즈, 앞뒤로 검토하는 문자개수
    word_min_count = 5   #최소 등장하는 문자의 개수
    sampling_threshold = 1e-5  #
    negative_size = 5       
    train_epoch = 100
    dm = 1 #0 = dbow; 1 = dmpv
    worker_count = cores #number of parallel processes

    #9. 크린파일 로드하기
    LabeledSentence = gensim.models.doc2vec.LabeledSentence

    #9. 크린파일 로드하기
#    sentences=doc2vec.TaggedLineDocument(save_clean_file)

    docs = []
    tags = []
    okt = Okt()
    
    with open(save_clean_file, encoding = 'utf-8') as data:
        r = csv.reader(data)
        print('크린파일 읽음')
        for i in r:
   #         words = word_tokenize(i[1])  한글지원하지 않음 오류
            words = okt.morphs(i[1], stem=True) # 토큰화
            tags.append(i[0])
            docs.append(words)
    print(tags[:10])
    print(docs[:10])
            
    #문자와 tags 데이터 라벨링
    it = LabeledLineSentence(docs, tags)
    ##모델셋팅
    print('모델링 준비')
    model = gensim.models.Doc2Vec(size=vector_size, window=window_size, min_count=word_min_count, workers=cores,alpha=0.025, min_alpha=0.025, iter=10)
    print('모델링 초기화 완료')
    model.build_vocab(it)
    print('사전 만들기 완료')
    print('모델 학습 시작')

    ##11. 모델 학습하기
    model.train(it, epochs=model.iter, total_examples=model.corpus_count)
#
#    for epoch in tqdm(range(10)):
#        model.train(it, epochs=model.iter, total_examples=model.corpus_count)
#        model.alpha -= 0.002 # decrease the learning rate
#       model.min_alpha = model.alpha # fix the learning r##ate, no deca
#        model.train(it)
    
    print('모델 학습 완료')
    
  #Most Similar Docs
 #   t = word_tokenize("안녕") #한글 지원하지 않음 오류
    t = okt.morphs("목소리 포스터", stem=True) # 토큰화

    tokens = model.infer_vector(t)
    sims = model.docvecs.most_similar([tokens])
    print(sims)

    ##11. 모델 학습하기
    # Train document vectors!
  #  for epoch in range(10):
  #      doc_vectorizer.train(sentences, epochs=model.iter, total_examples=model.corpus_count)
  #      doc_vectorizer.alpha -= 0.002 # decrease the learning rate
  #      doc_vectorizer.min_alpha = doc_vectorizer.alpha # fix the learning rate, no decay

    #12. 모델 저장하기 To save
    model.save(modelfile)
    model.save_word2vec_format(word2vec_file, binary=False)
    print('모델 저저장완료')


##################################################################
### 모델테스트 함수
##################################################################
#유사도 체크하는 함수
def BSdoc2vectest():
    global modelfile, word2vec_file
    
    model = gensim.models.Doc2Vec.load(modelfile)
    print('포스터: \n', model.wv.most_similar('포스터', topn=7),'\n')
   # print('송중기: \n',model.wv.most_similar('송중기', topn=7),'\n')
   # print('목소리: \n', model.wv.most_similar('목소리', topn=7),'\n')
   # print('포스터: \n', model.wv.most_similar('포스터', topn=7),'\n')
   # print('쓰레기: \n', model.wv.most_similar('쓰레기', topn=7),'\n')
   # print('+영화, +남자배우, -여배우 :\n', model.wv.most_similar(positive=['영화', '남자배우'], negative=['여배우'], topn=7),'\n')
    nsimilar=7
    while 1:
        wsimilar = input("유사도가 궁금한 단어 ? = ")
        try:
            nsimilar = int(input("출력개수는(99는끝남) ? = "))
        except ValueError:
            print("1~300 사이의 숫자")
        if nsimilar == 99:
            return
        #출력
        print('{0} 과 유사도가 높은 단어들 :\n'.format(wsimilar))
        try:
            result = model.wv.most_similar(wsimilar, topn=nsimilar)
        except KeyError:
            print("단어 정보가 없습니다.")
            return
        print(result)
        print('\n')


#$$$$$$$$$$$$$#####실행 시키기
##### 1. 데이터파일 읽어와서 크린데이터 파일에 모듈 콜하기
#make_CleanData()

###### 2. 크린데이터 불러와서 doc2vec 학습시키기
#BSdoc2vectrain()

####### 3. 모델테스트 유사도 체크
BSdoc2vectest()


