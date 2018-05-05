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
import re
root_dir = r"C:/Users/auas/Desktop/auas/大四/毕设/结题/code/key_word/"

class raw_corpus_loader(object):
    def __init__(self,file_path):
        self.file_path = file_path
        self.raw_data = open(self.file_path,"r",encoding="utf8").readlines()
        return
    def show_sample(self):
        print("there are {0} sentences: ",len(self.raw_data))
        for idx in range(3):
            print("@@@@@@@@@@@")
            print(self.raw_data[idx].split(" "))
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
    auas_loader = raw_corpus_loader(root_dir + "data/doc2vecall_corpus.txt")
    raw_data = auas_loader.show_sample()
    model_type = "advance"
    auas_crf_extractor = crf_key_extractor(model_type, model_path="")
    for idx in range(len(raw_data)):
        tmp_sent = raw_data[idx]
        y_pred, key_lst = auas_crf_extractor.extract_sent2keyword(tmp_sent)
        ret = ret + [it[0] for it in key_lst]
        if idx % 50 == 0:
            print("@@@@")
            pprint(tmp_sent)
            print("iter: ",idx)
            print(key_lst)
    f = open(save_path,"w",encoding="utf8")
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
    tmp_ret = list(set(tmp_ret))
    for it in tmp_ret:
        f.writelines(it+"\n")
    f.close()
    print("there are total key_words: ",len(tmp_ret))

save_path = "key_word.txt"
save_key_words(save_path)





