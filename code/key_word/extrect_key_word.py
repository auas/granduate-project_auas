'''
this file use crf or lstm/crf model to extract key word
key word is a list, each item is a key word
generated key word will save as "key_word_{method}.pkl"
overall this task separated into 3 parts
(1) extact raw data from file: write in a txt file:
each line is a sent and is separated by " "
(2) modify trained model to give prediction for each word in every sentences
(3) gether all the key words and save them in the file
'''
import jieba
import json
from auas_CRF import auas_CRF
from pprint import pprint as pprint
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import operator
import math
root_dir = r"C:/Users/auas/Desktop/auas/大四/毕设/结题/code/key_word/"
data_root =  r"C:/Users/auas/Desktop/auas/大四/毕设/结题/data/"
class raw_corpus_loader(object):
    def __init__(self,file_path):
        self.file_path = file_path
        f = open(self.file_path,"rb")
        self.pre_raw_data = pickle.load(f)
        f.close()
        self.JD = self.pre_raw_data["JD"]
        self.JL = self.pre_raw_data["JL"]
        self.punc = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~。、]+'
        self.sep_pattern = "(\d\.|\t|\d、|\.|,|，|。|;|；)"
        self.raw_data = []
        for jd in self.JD:
            self.raw_data = self.raw_data + self.JDJL2text(jd)
        for jl in self.JL:
            self.raw_data = self.raw_data + self.JDJL2text(jl)
        self.raw_data = [it for it in self.raw_data if not (it == " " or it == "")]
        return
    def clear_sent(self,sent):
        # print("!!!")
        # print(sent)
        ret = re.sub(self.sep_pattern," ",sent)
        # print(ret)
        ret = re.sub(self.punc," ",ret)
        # print(ret)
        ret.replace("／"," ")
        ret = re.sub("\s+"," ",ret)
        # print(ret)
        return ret.lower()
    def JDJL2text(self,a_jd):
        ret = re.split(self.sep_pattern,a_jd)
        ret = [self.clear_sent(it) for it in ret]
        print("~~~~~~~~~~")
        print(ret)
        return ret

    def show_sample(self):
        print("there are {0} sentences: ",len(self.raw_data))
        for idx in range(30):
            print(self.raw_data[idx])
            print("##################")
        return self.raw_data

class raw_corpus_loader_51job(object):
    def __init__(self,file_path):
        self.file_path = file_path
        f = open(self.file_path,"rb")
        self.pre_raw_data = pickle.load(f)
        f.close()
        self.JL = self.pre_raw_data
        self.JD = []
        self.punc = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~。、]+'
        self.sep_pattern = "(\d\.|\t|\d、|\.|,|，|。|;|；)"
        self.raw_data = []
        for jd in self.JD:
            self.raw_data = self.raw_data + self.JDJL2text(jd)
        for jl in self.JL:
            self.raw_data = self.raw_data + self.JDJL2text(jl)
        self.raw_data = [it for it in self.raw_data if not (it == " " or it == "")]
        return
    def clear_sent(self,sent):
        # print("!!!")
        # print(sent)
        ret = re.sub(self.sep_pattern," ",sent)
        # print(ret)
        ret = re.sub(self.punc," ",ret)
        # print(ret)
        ret.replace("／"," ")
        ret = re.sub("\s+"," ",ret)
        # print(ret)
        return ret.lower()
    def JDJL2text(self,a_jd):
        ret = re.split(self.sep_pattern,a_jd)
        ret = [self.clear_sent(it) for it in ret]
        print("~~~~~~~~~~")
        print(ret)
        return ret

    def show_sample(self):
        print("there are {0} sentences: ",len(self.raw_data))
        for idx in range(30):
            print(self.raw_data[idx])
            print("##################")
        return self.raw_data


class key_extractor(object):
    def __init__(self,model_path,model_type):
        self.model_path = model_path
        self.model_type = model_type
        self.load_model(model_path,model_type)
        return
    def load_model(self,model_path,model_type):
        print("loading model for: ",model_type)
        raise NotImplemented
    def check_sent_format(self,sent):
        if isinstance(sent,str):
            tmp_sent = list(sent.split(" "))
        elif isinstance(sent,list):
            tmp_sent = sent
        else:
            raise ValueError("sent type not supported : {0}".format(type(sent)))
        pre_sent = jieba.posseg.cut(" ".join([it for it in tmp_sent if not it == "\n"]))
        new_sent = [(i.word, i.flag) for i in pre_sent if not i.word == " "]
        return new_sent
    def extract_sent2keyword(self,sent):
        raise NotImplemented


class crf_key_extractor(key_extractor):
    def __init__(self,model_type,model_path = ""):
        super(crf_key_extractor,self).__init__(model_path,model_type)
        # print("model_type: ",self.model_type)
        # print("model_path: ",self.model_path)
        self.load_model(model_type,model_path)
    def load_model(self,model_type,model_path = ""):
        self.CRF = auas_CRF()
        self.CRF.load_model(self.model_path,self.model_type)
        self.model = self.CRF.model
    def extract_sent2keyword(self,sent):
        sent = self.check_sent_format(sent)
        # print("@@@@@@@@@@@@@@@")
        # print("org_sent: ")
        org_sent = sent
        # pprint(org_sent)
        # print("@@@@@@@@@@@@@@@")
        if self.model_type == "advance":
            sent = [self.CRF.word2features(org_sent,idx) for idx in range(len(org_sent))]
            # pprint(sent)
        # print("predicting")
        y_pred = self.model.tag(sent)
        # print("y_pred: ",y_pred)
        key_lst = [org_sent[idx] for idx,it in enumerate(y_pred) if it == "1"]
        return y_pred,key_lst

# lower and return (maybe add more later)
def clear_key_word(w):
    ret = w.lower()
    return ret

def save_key_words(save_path):
    def clear_str(pre_str):
        ret_str = re.sub("(\\|\n|\(|\)|\）|\（|)"," ",pre_str)
    ret = []
    # auas_loader = raw_corpus_loader(root_dir + "data/extracted_text.pkl")
    auas_loader = raw_corpus_loader_51job(root_dir + "data/51job.pkl")
    raw_data = auas_loader.show_sample()
    model_type = "advance"
    auas_crf_extractor = crf_key_extractor(model_type, model_path="")
    for idx in range(len(raw_data)):
        tmp_sent = raw_data[idx]
        y_pred, key_lst = auas_crf_extractor.extract_sent2keyword(tmp_sent)
        ret = ret + [it[0] for it in key_lst]
        if idx % 100 == 0:
            print("@@@@")
            pprint(tmp_sent)
            print("iter: ",idx)
            print(key_lst)
    f = open(save_path,"wb")
    tmp_ret = []
    dump_lst = ["-","\\","1",""]
    for it in ret:
        it = clear_key_word(it)
        if len(it)>20:
            continue
        if it.isdigit():
            continue
        if it in dump_lst:
            continue
        tmp_ret.append(it)
    ret = {}
    for key_w in tmp_ret:
        if key_w not in ret:
            ret[key_w] = 0
        ret[key_w] = ret[key_w] + 1
    pickle.dump(ret,f)
    f.close()
    print("there are total key_words: ",len(ret))

# save_path = "key_word_51job.pkl"
# save_key_words(save_path)

class doc2key_vec(object):
    def __init__(self,keyword_path):
        self.keyword = list(open(keyword_path,"r",encoding="utf8").readlines())
        self.keyword = [it.replace("\n","") for it in self.keyword if self.chear_keyword(it)]
        # print(self.keyword)
        # exit(555)
        self.keyword_len = len(self.keyword)
        print("there are {0} key words in total ".format(self.keyword_len))
        self.punc = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~。、]+'
        self.sep_pattern = "(\d\.|\t|\d、|\.|,|，|。|;|；)"
        return
    def clear_doc(self,doc):
        ret = re.sub(self.sep_pattern," ",doc)
        # print(ret)
        ret = re.sub(self.punc," ",ret)
        # print(ret)
        ret.replace("／"," ")
        ret = re.sub("\s+"," ",ret)
        # print(ret)
        return ret.lower()
    def chear_keyword(self,k):
        if k == " ":
            return False
        if k == "\\":
            return False
        return True
    def fenci(self,doc):
        ret = list(jieba.cut(doc))
        return ret
    def is_keyword(self,word):
        for k in self.keyword:
            if self.is_same(word,k):
                return True
        return False
    def is_same(self,w1,w2):
        if w1 == w2:
            return True
        else:
            return False
    def doc2keyword(self,doc):
        doc = self.clear_doc(doc)
        word_list = self.fenci(doc)
        # print("~~")
        # print(word_list)
        # print("~~")
        return list(set([it for it in word_list if self.is_keyword(it)]))
    def word2id(self,w):
        lst = [idx for idx,k in enumerate(self.keyword) if self.is_same(w,k)]
        return lst[0]
    def doc2vec(self,doc):
        tmp_k = self.doc2keyword(doc)
        # print(doc)
        print("tmp_k: ",tmp_k)
        idx_lst = [self.word2id(w) for w in tmp_k]
        if len(tmp_k) == 0:
            print("cannot find key word in doc: ",doc)
            print("@@@@@@@@@@@@@@@@@@@@")
        ret = np.zeros(self.keyword_len)
        for idx in idx_lst:
            ret[idx] = 1
        return ret

def load_JDJL(path):
    r = pickle.load(open(path,"rb"))
    JD = r["JD"]
    JL = r["JL"]
    return [JD,JL]

def save_onehot_vec(spath = root_dir+"/data/onehot_vec_matched.pkl"):
    d = []
    JD,JL = load_JDJL(root_dir+"/data/extracted_text.pkl")
    # JD = []
    # f = open(root_dir+"/51job.pkl","rb")
    # JL = pickle.load(f)
    # f.close()
    keyword_path = root_dir + "key_word.txt"
    auas_doc2vec = doc2key_vec(keyword_path)
    for idx,a_jl in enumerate(JL):
        if idx%200 == 0:
            print("iter: ",idx)
        vec = auas_doc2vec.doc2vec(a_jl)
        d.append(vec)
    sf = open(spath,"wb")
    pickle.dump(d,sf)
# save_onehot_vec()

def check_save_key_word():
    p = "key_word_51job.pkl"
    f = open(p,"rb")
    d = pickle.load(f)
    f.close()
    sd = sorted(d.items(), key=operator.itemgetter(1))
    num_arr = np.array([math.log(d[it],10) for it in d ]) #if not d[it] == 1
    def draw_hist(arr):
        plt.clf()
        plt.hist(num_arr,bins = 200)
        plt.title("key word counts")
        plt.ylabel("counts")
        plt.xlabel("log hit numbers")
        plt.xlim(-0.1,num_arr.max()+1)
        plt.savefig("keyword_count.png")
        # plt.show()
    # draw_hist(num_arr)
    print("saving key word")
    # f = open("key_word.txt","w",encoding="utf8")
    min_frq = 3
    save_lst = [it for it in sd]
    c = 0
    for w_idx in range(len(save_lst)):
        w = save_lst[len(save_lst)-w_idx-1]
        if w[1] > min_frq:
            c = c+1
            # f.write(w[0] + "\n")
    # f.close()
    print(c)


# check_save_key_word()

def save_onehot_vec_final(spath = root_dir+"/data/onehot_vec_matched_final.pkl"):
    f = open(data_root+"sorted_data_final.pkl","rb")
    raw_data = pickle.load(f)
    f.close()
    keyword_path = root_dir + "key_word.txt"
    auas_doc2vec = doc2key_vec(keyword_path)
    x = []
    y = []
    jd = []
    for data_idx in raw_data:
        JL = raw_data[data_idx]["JL"]
        a_jd = raw_data[data_idx]["JD"]
        vec_jd = auas_doc2vec.doc2vec(a_jd)
        for idx,a_jl in enumerate(JL):
            if idx%200 == 0:
                print("iter: ",idx)
            vec = auas_doc2vec.doc2vec(a_jl)
            jd.append(vec_jd)
            x.append(vec)
            y.append(data_idx)
    sf = open(spath,"wb")
    s_data = {}
    s_data["JL"] = np.array(x)
    s_data["JD"] = np.array(jd)
    s_data["y"] = np.array(y)
    pickle.dump(s_data,sf)

save_onehot_vec_final()