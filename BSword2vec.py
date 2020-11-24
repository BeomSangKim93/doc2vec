#pasas 사용 import
import pandas as pd
from BSword2vecClass import BSword2vecUtility

#3가지 데어터 로딩 구문
train = pd.read_csv('data/labeledTrainData.tsv', 
                    header=0, delimiter='\t', quoting=3)
test = pd.read_csv('data/testData.tsv', 
                   header=0, delimiter='\t', quoting=3)
unlabeled_train = pd.read_csv('data/unlabeledTrainData.tsv', 
                              header=0, delimiter='\t', quoting=3)

#읽어들인 형태 인쇄
print(train.shape)
print(test.shape)
print(unlabeled_train.shape)

#'review'크기 인쇄
print(train['review'].size)
print(test['review'].size)
print(unlabeled_train['review'].size)