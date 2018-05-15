'''
this file sort the matched data in different class
save file in .pkl file.
ret = {class_id:{JD:"",JL:[]}}
'''
import pickle
import os
import glob
import numpy
import zipfile
import re
import pickle
import jieba
from pprint import pprint as pprint
root_dir = r"C:/Users/auas/Desktop/auas/大四/毕设/结题/code/key_word/"
data_root =  r"C:/Users/auas/Desktop/auas/大四/毕设/结题/data/"

def get_sorted_data_path():
    data_dir = data_root + "JD2JL/"
    dir_lst = glob.glob(data_dir+"*")
    print(dir_lst)
    ret = {}
    for idx,item in enumerate(dir_lst):
        ret[idx] = {}
        ret[idx]["JD"] =  glob.glob(item + "/*.txt")[0]
        tmp_JL_zip = glob.glob(item + "/*.zip")
        tmp_JL_p = []
        for item in tmp_JL_zip:
            print(item)
            zipf = zipfile.ZipFile(item)
            new_dir = item.replace(".zip", "")
            zipf.extractall(new_dir)
            tmp = []
            tmp = tmp + glob.glob(new_dir + "/*")
            tmp_1 = []
            for it in tmp:
                tmp_1 = tmp_1 + glob.glob(it + "/*")
            for it in tmp_1:
                tmp_JL_p = tmp_JL_p + glob.glob(it + "/*")
        ret[idx]["JL"] = tmp_JL_p

    return ret

class JD(object):
    def __init__(self,path):
        f = open(path,"r",encoding="utf8")
        self.raw_data = f.readlines()
        return
    def show_raw_data(self):
        for line in self.raw_data:
            print(line)
        return
    def extra_text_data(self):
        tmp = []
        for it in self.raw_data:
            it.replace("\n"," ").replace("\t"," ")
            tmp = tmp + it.split("\n")
        tmp = [it for it in tmp if not (it == "" or it == " ")]
        for idx,it in enumerate(tmp):
            print(idx,"  ",it)
        return " ".join(tmp)
class JL(object):
    def __init__(self,path):
        f = open(path,"r",encoding="utf8")
        self.raw_data = f.readlines()
        return
    def show_raw_data(self):
        for idx,line in enumerate(self.raw_data):
            print(idx,"  ",line)
        return
    def clear_text(self,str):
        str = str.replace("\n"," ").replace("\t"," ").replace("-"," ").replace("/"," ")
        str = re.sub("\d、|/(|/)|）|（|、|，|。|,|\.|\:|：|；"," ",str)
        lst_a = re.split("<(.*?)>",str)
        lst_b = re.findall("<(.*?)>", str)
        # print("str: ",str)
        # print("lst_a: ",lst_a)
        # print("lst_b: ",lst_b)
        lst = [it for it in lst_a if it not in lst_b and not it == ""]
        ret_str = " ".join(lst)
        ret_str = re.sub(r'\s+', ' ', ret_str)
        # print(ret_str)
        # exit(0)
        return " ".join(lst)
    def extra_text_data(self):
        ret = []
        got_duty = 0
        key_word_lst = ["职责业绩","项目职责","项目业绩","项目简介"]

        for idx,line in enumerate(self.raw_data):
            j = False
            for key_word in key_word_lst:
                if key_word in line:
                    j = True
                    break
            if j:
                got_duty = 1
                tmp_duty = ""
            if got_duty ==1:
                tmp_duty = tmp_duty+line
            if "</tr>" in line and got_duty == 1:
                got_duty = 0
                ret.append(self.clear_text(tmp_duty))
                tmp_duty = ""
        pre_r = " ".join(ret)
        r = re.sub("\s+"," ",pre_r)
        return r
rp = get_sorted_data_path()
r = {}
for data_idx in rp.keys():
    JD_p = rp[data_idx]["JD"]
    JL_p_lst = rp[data_idx]["JL"]
    r[data_idx] = {}
    r[data_idx]["JD"] = JD(JD_p).extra_text_data()
    # p = JL_p_lst[0]
    # r = JL(p).extra_text_data()
    # print(r)
    # print(len(r))
    # pprint(JL_p_lst)
    # exit(111)
    r_lst = [JL(p).extra_text_data() for p in JL_p_lst]
    pprint(r_lst)
    r[data_idx]["JL"] = r_lst
f = open(data_root+"sorted_data_final.pkl","wb")
pickle.dump(r,f)
f.close()
pprint(r)

