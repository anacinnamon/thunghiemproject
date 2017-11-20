from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
from nltk import  word_tokenize
import io
import re
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
import csv
from sklearn.metrics import accuracy_score

def readCSV(csvfilename):
    mylist = []

    with open(csvfilename, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            mylist.append(row)

    return mylist

#xay dung tu dien tieng viet
def buildVietnameseVocabulary():
    vocabulary = []
    tokenizer = RegexpTokenizer(r'\w+')

    with io.open("VNTQcorpus-small.txt",'r',encoding='utf8') as f:
        text = f.read()
        xx = set(tokenizer.tokenize(text.lower()))
        for item in xx:
            item = re.sub("\d+", "", item)
            if item != "":
                vocabulary.append(item)

    #print(len(vocabulary))
    return vocabulary

# tien xu ly ( chinh ta - lemma - chu viet hoa - gom nhom - remove strange character , tf - idf , count or binary ) , thu nghiem voi ngram, thu nghiem voi cac thuat toan khac nhau
def preprocessingData(file_train, vocabulary):
    # corpus_first = ['đâ12y là! , co1n 123chó...','đây là!! con... 1234 mèo123 mèo','nó là@ con khỉ khỉ']
    corpus_first = []
    s1 = ""
    s2 = ""
    s3 = ""
    X = []
    X_label = []
    for i in range(len(file_train)):
        flag = True
        comment = file_train[i][2]
        label = file_train[i][8]
        if label == "-1":
            s1 = s1 + " " + comment
        elif label == "0":
            s2 = s2 + " " + comment
        elif label == "1":
            s3 = s3 + " " + comment
        else:
            print("label wrong at "+str(i))
            flag = False
        if flag == True:
            X.append(comment)
            X_label.append(label)

    corpus_first.append(s1)
    corpus_first.append(s2)
    corpus_first.append(s3)

    # for item in corpus_first:
    #     print(item)
    tokenizer = RegexpTokenizer(r'\w+')
    corpus = []
    for item in corpus_first:
        itemtoken = tokenizer.tokenize(item.lower())
        for i in range(len(itemtoken)):
            itemtoken[i] = re.sub("\d+", "", itemtoken[i])

        j = 0
        while (j < len(itemtoken)):
            if (itemtoken[j] not in vocabulary) or (itemtoken[j][0].isupper()):
                itemtoken.remove(itemtoken[j])
            else:
                j += 1

        corpus.append(" ".join(itemtoken))

    # for item in corpus:
    #     print(item)
    return corpus,X,X_label

# xay dung best features tf-idf
def getBestFeatures(corpus):
    vectorizer = TfidfVectorizer()
    tftfidf = vectorizer.fit_transform(corpus)

    top = 10
    list_feature = []

    for k in range(len(corpus)):
        listtup1 = []
        group1 = tftfidf.toarray()[k]
        index = 0
        for item in group1:
            tup =(item,index)
            listtup1.append(tup)
            index+=1
        sorted_by_second = sorted(listtup1, key=lambda tup: tup[0],reverse=True)
        for i in range(top):
            list_feature.append(vectorizer.get_feature_names()[sorted_by_second[i][1]])

    list_feature = list(set(list_feature))
    # print(list_feature)
    return list_feature

def buildVectorTrainData(list_feature,list_train_data,label_train):
    vectorizer = CountVectorizer(vocabulary=list_feature)
    array_train_data = vectorizer.transform(list_train_data).toarray()
    # print(vectorizer.get_feature_names())
    train_data = []

    for x in range(len(list_train_data)):
        vect = {}
        for y in range(len(list_feature)):
            vect[list_feature[y]] = array_train_data[x][y]
        tup1 = (vect, label_train[x])
        train_data.append(tup1)

    #print(train_data)
    return train_data

def buildVectorTestData(list_feature,list_test_data):
    vectorizer = CountVectorizer(vocabulary=list_feature)
    array_test_data = vectorizer.transform(list_test_data).toarray()
    test_data = []

    for x in range(len(list_test_data)):
        vect = {}
        for y in range(len(list_feature)):
            vect[list_feature[y]] = array_test_data[x][y]
        tup1 = (vect)
        test_data.append(tup1)

    #print(test_data)
    return test_data


if __name__ == '__main__':

    # du an phan loai comments
    # viet thuong , xoa dau cau , xoa number

    # doc file train.csv
    file_train = readCSV("comment_label_test.csv")

    # build vocabulary
    vocabulary = buildVietnameseVocabulary()

    # tien xu ly du lieu
    corpus,X,X_label = preprocessingData(file_train,vocabulary)

    # for item in corpus:
    #     print(item)


    # get best features
    list_feature = getBestFeatures(corpus)
    print(list_feature)

    # K-fold to test
    list_train_data = []
    list_test_data = []
    label_train = []
    label_test = []
    k_fold = KFold(n_splits=3)
    for train_indices, test_indices in k_fold.split(X):
        print("train")
        for item in train_indices:
            list_train_data.append(X[item])
            label_train.append(X_label[item])
        print("test")
        for item in test_indices:
            list_test_data.append(X[item])
            label_test.append(X_label[item])
        break

    print(list_train_data)
    print(label_train)
    print(list_test_data)
    print(label_test)

    # bieu dien vector input
    train_data = buildVectorTrainData(list_feature, list_train_data, label_train)
    # example train_data = [({"stupid": 0, "lovely": 1, "dog": 2,"cat":0}, "positive_dog"),
    #               ({"stupid": 1, "lovely": 0, "dog": 0, "cat": 2}, "negative_cat"),
    #               ({"stupid": 0, "lovely": 0, "dog": 0, "cat": 0}, "normal")]
    test_data = buildVectorTestData(list_feature,list_test_data)

    # models and measure
    classif = SklearnClassifier(BernoulliNB()).train(train_data)
    # classif = SklearnClassifier(SVC(C=1.0, kernel='rbf', degree=3), sparse=False).train(train_data)
    # measure accuracy
    y_pred = classif.classify_many(test_data)
    y_true = label_test
    print(accuracy_score(y_true, y_pred))
    # y_true = [0, 1, -1, -1, 0]
    # y_pred = [0, 0, -1, 1, 0]
    # target_names = ['class 0', 'class 1', 'class 2']
    # print(classification_report(y_true, y_pred, target_names=target_names))