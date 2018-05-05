# -*- coding:utf-8 -*-
'''
CRF extract key word from the document
'''
from pprint import pprint as pprint
import os
from topology import key_word_extract_model
import jieba.posseg as pseg
import jieba
import pycrfsuite as crf
import pycrfsuite
import pickle
import re
import json
data_root = r"C:/Users/auas/Desktop/auas/大四/毕设/结题/data/"
save_root = r"C:/Users/auas/Desktop/auas/大四/毕设/结题/key_word/"

class auas_CRF(key_word_extract_model):
    def __init__(self):
        super(auas_CRF,self).__init__("CRF")
        print("initiated")
        return
    def load_data(self,data_path,model_type):
        super(auas_CRF,self).load_raw_data(data_path)
        if model_type == "baseline":
            return
        elif model_type == "advance":
            self.training_set = self.advance_feature_extract(self.training_set)
            self.test_set = self.advance_feature_extract(self.test_set)

    def word2features(self,sent, i):

        word = sent[i][0]
        postag = sent[i][1]

        # print("!!!!!!!!!!!!!")
        # print("word",word)
        # print("postag",postag)
        features = [
            'bias',
            'word.lower=' + word.lower(),
            'word[-3:]=' + word[-3:],
            'word[-2:]=' + word[-2:],
            'word.isupper=%s' % word.isupper(),
            'word.istitle=%s' % word.istitle(),
            'word.isdigit=%s' % word.isdigit(),
            'postag=' + postag,
            'postag[:2]=' + postag[:2],
        ]
        if i > 0:
            word1 = sent[i - 1][0]
            postag1 = sent[i - 1][1]
            features.extend([
                '-1:word.lower=' + word1.lower(),
                '-1:word.istitle=%s' % word1.istitle(),
                '-1:word.isupper=%s' % word1.isupper(),
                '-1:postag=' + postag1,
                '-1:postag[:2]=' + postag1[:2],
            ])
        else:
            features.append('BOS')

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            postag1 = sent[i + 1][1]
            features.extend([
                '+1:word.lower=' + word1.lower(),
                '+1:word.istitle=%s' % word1.istitle(),
                '+1:word.isupper=%s' % word1.isupper(),
                '+1:postag=' + postag1,
                '+1:postag[:2]=' + postag1[:2],
            ])
        else:
            features.append('EOS')
        # print("@@@@@@@@@@@@")
        # print("word",word)
        # print("postag",postag)
        # print("features",features)
        return features
    def load_model(self,model_path,model_type):
        self.model = pycrfsuite.Tagger()
        if model_type == "baseline":
            self.model.open(model_path + 'crf_baseline.model')
        elif model_type == "advance":
            self.model.open(model_path + 'crf_advance.model')
        else:
            raise ValueError("not support model type: {0}".format(model_type))
        return
    def test_model(self,data_path,model_path,model_type):
        self.load_model(model_path,model_type)
        # self.load_raw_data(data_path)
        self.load_data(data_path,model_type)
        y = [it["tag"] for it in self.test_set]
        y_pred = [self.model.tag(it["word"]) for it in self.test_set]
        return self.show_result(y_pred,y)
    def advance_feature_extract(self,raw_dataset):

        r = []
        for sent in raw_dataset:
            tmp_dic = {}
            tmp_dic["word"] = [self.word2features(sent["word"], idx) for idx in range(len(sent["word"]))]
            tmp_dic["tag"] = sent["tag"]
            r.append(tmp_dic)
        return r
    def train_model(self,data_path,save_dir,model_type):
        self.load_data(data_path,model_type)
        print("training CRF: ",model_type)
        self.crf = crf.Trainer()
        print("loading training data")
        for it in self.training_set:
            self.crf.append(it["word"], it["tag"])
        self.crf.set_params({
            # coefficient for L1 penalty
            'c1': 0.1,
            # coefficient for L2 penalty
            'c2': 0.01,
            # maximum number of iterations
            'max_iterations': 200,
            # whether to include transitions that
            # are possible, but not observed
            'feature.possible_transitions': True
        })
        self.crf.train(save_dir + 'crf_{0}.model'.format(model_type))
        return

def clear_word(word):
    # re_pattern = ",|.|:|;|。|，|：|；"
    ret = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",word)
    return ret

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')

    return features

def sort_raw_training_data(data_path):
    def load_raw_data(data_path):
        '''
        give a data path return the training and test data for crf
        :param data_path:
        :return: a list, each item is a dict {"tag","word"}
        '''
        json_data = open(data_path, "r", encoding="utf8")
        rr1 = json_data.readlines()
        rr2 = []
        rr3 = []
        for it in rr1:
            tmp_lst = [item for item in it.split(" ") if not item == ""]
            rr2.append(tmp_lst)
        for it in rr2:
            tmp_lst = [it.replace("\"", "").replace("\n", "") for it in it]
            rr3.append(tmp_lst)
        rr4 = []
        tmp_dic = {}
        got_word = 0
        for idx, it in enumerate(rr3):
            if it[0] == "_id":
                if idx == 0:
                    continue
                else:
                    rr4.append(tmp_dic)
                tmp_dic = {}
                tmp_dic["word"] = []
                tmp_dic["tag"] = []
            if it[0] == "word":
                tmp_dic["word"].append(it[2])
                got_word = 1
            if it[0] == "tag" and got_word == 1:
                got_word = 0
                tmp_dic["tag"].append(it[2])
        raw_data = rr4[1:]
        return raw_data
    raw_data = load_raw_data(data_path)
    ret = []
    for sent_idx,item in enumerate(raw_data):
        sent = item["word"]
        sent = [clear_word(it) for it in sent if not clear_word(it) == ""]
        # pprint(sent)
        # print("!!")
        tag = item["tag"]
        str_sent = " ".join(sent)
        pre_sent = jieba.posseg.cut(str_sent)
        new_sent = [(i.word, i.flag) for i in pre_sent if not i.word == " "]
        tmp_dic = {}
        tmp_dic["word"] = []
        tmp_dic["tag"] = []
        for it in new_sent:
            for word_idx,word in enumerate(item["word"]):
                # print(word_idx,word)
                if it[0] == clear_word(word):
                    tmp_dic["word"].append(it)
                    tmp_dic["tag"].append(item["tag"][word_idx])
        ret.append(tmp_dic)

        if sent_idx % 100 == 0:
            print("sent_idx: ",sent_idx)
            pprint(tmp_dic)
            print("@@@@@@@@@@@@@@@")


        # print("pre_len: ",len(sent))

        # print("new_len ",len(new_sent) )
        # pprint(new_sent)
        # print("sorted")
        # pprint(tmp_dic)
        # exit(0)
    return ret

    # return raw_data[0]

auas = 1
# f = open("sorted_crf_data.pkl","w",encoding="utf8")
# data_path = data_root + "extraction.corpus_all.json"
# r = sort_raw_training_data(data_path)
# str = json.dumps(r)
# f.write(str)
# f.close()

# print("@@@@@@@@@@")
# print(sent)
# print("@@@@@@@@@@")
# feature = [word2features(sent, idx) for idx in range(len(sent))]
# pprint(pprint(feature))
crf_training_data_path = data_root + "sorted_crf_data.pkl"
# tst = auas_CRF()
# tst.train_model(crf_training_data_path,"","baseline")
# tst.train_model(crf_training_data_path,"","advance")
# print("CRF baseline")
# tst.test_model(crf_training_data_path,"","baseline")
# pprint("@@@@@")
# print("CRF consider advanced")
# tst.test_model(crf_training_data_path,"","advance")