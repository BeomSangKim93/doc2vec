# 출력이 너무 길어지지 않게하기 위해 찍지 않도록 했으나
import warnings
import pandas as pd
import numpy as np
import csv
import time
import os
# 형태소 분석기 로딩
from konlpy.tag import Okt

# gensim의 doc2vec 이용
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# from nltk.tokenize import word_tokenize
# from gensim.models import doc2vec
from multiprocessing import Value, Process, Manager
import multiprocessing
# from gensim.models.doc2vec import TaggedDocument
import gensim

warnings.filterwarnings('ignore')

LabeledSentence = gensim.models.doc2vec.LabeledSentence

# 파일이름 세팅하기
DATA_PATH = 'data/'
MODEL_PATH = 'models/'
load_train_file = DATA_PATH + 'ratings_train.txt'
load_test_file = DATA_PATH + 'ratings_test.txt'
save_clean_file = DATA_PATH + 'clean_naver_okt.csv'
modelfile = MODEL_PATH + 'doc2vec.model'
word2vec_file = modelfile + '.word2vec_format'

# 글로벌 변수
stopwords = {'의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를',
                 '으로', '자', '에', '와', '한', '하다'}


# ############################################
# I. 전처리 후 파일 생성
# #############################################


# 토큰화 속도 개선을 위해 멀티프로세싱을 위해 함수정의
def MultiprocessTokenizeing(t, n_train, train_cnt):
    global stopwords

    okt = Okt()
    t_token = []

    # 5. 형태소 분석, DOC2VEC를 학습시키기 위해서는 Tag값이 필요하다
    t_token = okt.morphs(t[1], stem=True)  # 토큰화 t의 2번째에 문자열
    # 6. 불용어 제거
    t_token = [word for word in t_token if not word in stopwords]
    tags = t[0]
    t_token = [t_token, str(tags)]

    if (n_train+1) % 10000 == 0:
        print('문서 전처리 {} of {} / ProcesId {}'.format(n_train+1, train_cnt, os.getpid()))

    return t_token

# 토큰화 해서 파일에 저장하는 루틴, MultiprocessTokenizeing를 콜함
def make_CleanData():

    global DATA_PATH, MODEL_PATH, load_train_file, load_test_file
    global save_clean_file, stopwords
    global train_cnt

    # 네이버 영화 리뷰 데이터를 읽어 온다.
    kr_train = pd.read_csv(load_train_file, header=0, delimiter='\t',
                           quoting=3)

    # 네이버 영화 읽오왔는지 확인
    print(kr_train['document'].size, ' 개수의 데이터 파일 읽어옴')

    # 1. 중복 데이터 제거
    kr_train.drop_duplicates(subset=['document'], inplace=True)

    # 2.정규 표현식을 통한 한글 외 문자 제거
    kr_train['document'] = kr_train['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]",
                                                            "")
    # 3.빈내용은 Null로 표기
    kr_train['document'].replace('', np.nan, inplace=True)

    # 4. Null 값 제거
    kr_train = kr_train.dropna(how='any')
    # print(kr_train.isnull().values.any())  # Null 값이 존재하는지 확인

    # 4.1 트레이닝 개수 셋팅
    train_cnt = kr_train['document'].size

    # 형태소 분석기 OKT를 이용해서 형태로 분석
    # 한글 형태소 분석기로 분석하고, 불용어 제거하고 2단계로 영어보다 단순함.

    df = pd.DataFrame(kr_train)
    kr_train_list = df.values.tolist()  # padaa를 리스트로 변환

    # print(kr_train_list[0:20])

    print('문서 전처리 시작')
    start_time = int(time.time())

    # 멀티프로세서 진행상황 체크를 위해 값 최기화
    m = Manager()
    tokenized_data = m.list()  # 멀티프로세스 결과값
    train_cnt = kr_train['document'].size

    # 프로세스 개수리턴
    cores = multiprocessing.cpu_count()
    print('{}개수의 프로세스로 병렬처리'.format(cores))

    # 멀티프로세스 콜하는 루틴, 리스트에서 하나의 데이터씩 전달
    pool = multiprocessing.Pool(cores)
    tokenized_data = pool.starmap(MultiprocessTokenizeing, [(t, i, train_cnt)  for i, t in enumerate(kr_train_list)])
    pool.close()
    pool.join()
  
    print('토큰화 총 소요 시간 : {}'.format(time.time() - start_time))

    # 7. 크린 데이터 파일로 저장, 한줄씩 문자열로
    # $$$$ 문자열로 변환하여 한줄씩 저장
    # 토큰 문자열로 변환하고 한줄씩 추가
    # clean_train_reviews  파일로 저장
    with open(save_clean_file, 'wt', encoding='utf-8') as fp:
        for line in tokenized_data:
            temp_str = str('MOVIE_'+line[1]+',')
            temp_str += ' '.join(line[0])
            temp_str += '\n'
            fp.writelines(temp_str)


#################################################################
# ## 크린데이터 트레이닝 하기
################################################################

class LabeledLineSentence(object):
    global LabeledSentence

    def __init__(self, docs, tags):
        self.labels_list = tags
        self.doc_list = docs

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(words=doc, tags=[self.labels_list[idx]])


###########################################################
# II. doc2vec 학습 루틴
#########################################################


def BSdoc2vectrain():
    global modelfile, word2vec_file
    cores = multiprocessing.cpu_count()

    # 8. Gensim로딩하고 doc2vec 초기셋팅하기
    # doc2vec parameters
    vector_size = 300   # 벡터 사이즈
    window_size = 15    # 윈도우 사이즈, 앞뒤로 검토하는 문자개수
    word_min_count = 5   # 최소 등장하는 문자의 개수
    sampling_min = 0.025
    # negative_size = 5
    train_epoch = 20
    # dm = 1  # 0 = dbow; 1 = dmpv

    # 9. 크린파일 로드하기
    # LabeledSentence = gensim.models.doc2vec.LabeledSentence
    # sentences=doc2vec.TaggedLineDocument(save_clean_file)

    docs = []
    tags = []
    okt = Okt()

    with open(save_clean_file, encoding='utf-8') as data:
        r = csv.reader(data)
        print('크린파일 읽기 시작')
        for i in r:
            words = okt.morphs(i[1], stem=True)  # 토큰화
            tags.append(i[0])
            docs.append(words)
    print(len(docs), ' 개수의 데이터 파일 읽어옴')
    print('초기 2개 문서번호 데이터 ',tags[:2])
    print('초기 2개 워드데이터 ',docs[:2])

    # 문자와 tags 데이터 라벨링
    it = LabeledLineSentence(docs, tags)
    # #모델셋팅
    print('모델링 준비')
    model = gensim.models.Doc2Vec(size=vector_size, window=window_size,
                                  min_count=word_min_count, workers=cores,
                                  alpha=0.025, min_alpha=sampling_min, iter=train_epoch)
    print('모델링 초기화 완료')
    model.build_vocab(it)
    print('사전 만들기 완료')
    print('모델 학습 시작')

    # #11. 모델 학습하기
    model.train(it, epochs=model.iter, total_examples=model.corpus_count)

    print('모델 학습 완료')
    # Most Similar Docs
    # t = word_tokenize("안녕") #한글 지원하지 않음 오류
    t = okt.morphs("목소리 포스터", stem=True)  # 토큰화

    tokens = model.infer_vector(t)
    sims = model.docvecs.most_similar([tokens])
    print(sims)

    # #11. 모델 학습하기
    # Train document vectors!

    # 12. 모델 저장하기 To save
    model.save(modelfile)
    model.save_word2vec_format(word2vec_file, binary=False)
    print('모델 저저장완료')


##################################################################
# ## 모델테스트 함수
##################################################################
# 유사도 체크하는 함수
def BSdoc2vectest():
    global modelfile, word2vec_file

    model = gensim.models.Doc2Vec.load(modelfile)
    print('검색 예) 포스터: \n', model.wv.most_similar('포스터', topn=7), '\n')
    nsimilar = 7
    while 1:
        wsimilar = input("유사도가 궁금한 단어 ? = ")
        try:
            nsimilar = int(input("출력개수는(99는끝남) ? = "))
        except ValueError:
            print("1~300 사이의 숫자")
        if nsimilar == 99:
            return
        # 출력
        print('{0} 과 유사도가 높은 단어들 :\n'.format(wsimilar))
        try:
            result = model.wv.most_similar(wsimilar, topn=nsimilar)
        except KeyError:
            print("단어 정보가 없습니다.")
            BSdoc2vectest()
        print(result)
        print('\n')


# #####################################
# 프로그램 메인
# ####################################
if __name__ == '__main__':

    # #### 1. 데이터파일 읽어와서 크린데이터 파일에 모듈 콜하기
    # make_CleanData()

    # ##### 2. 크린데이터 불러와서 doc2vec 학습시키기
    # BSdoc2vectrain()

    # ###### 3. 모델테스트 유사도 체크
    BSdoc2vectest()


