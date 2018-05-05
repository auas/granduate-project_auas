# -*- coding:utf-8 -*-
'''
this file extract data from file
including the matching of JD and JL
'''
import os
import glob
import zipfile
import re
import pickle
import jieba
# C:\Users\auas\Desktop\auas\大四\毕设\中期\zh-NER-TF-master\data_path
root_dir = r"C:/Users/auas/Desktop/auas/大四/毕设/结题/code/key_word/"
data_root =  r"C:/Users/auas/Desktop/auas/大四/毕设/结题/data/"
import pickle
# C:\Users\auas\Desktop\auas\大四\毕设\结题\code\key_word\data
# "新华社北京3月26日电  国家主席习近平3月26日就俄罗斯克麦罗沃市发生" \
# "重大火灾向俄罗斯总统普京致慰问电。习近平在慰问电中表示，惊悉贵国克麦" \
# "罗沃市发生火灾，造成重大人员伤亡和财产损失。我谨代表中国政府和中国人民，" \
# "并以我个人的名义，对所有遇难者表示沉痛的哀悼，向受伤者和遇难者家属致以深切的同情和诚挚的慰问。" \
# "同日，国务院总理李克强也就此向俄罗斯总理梅德韦杰夫致慰问电，向遇难者表" \
# "示深切哀悼，向遇难者家属致以诚挚慰问。"
# f = open(path,"rb")
# sf = open(spath,"wb")
# w = pickle.load(f)
# pickle.dump(w,sf,protocol=2)
# exit(0)


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
        return "".join(tmp)
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
        return ret

def get_JD_file():
    '''
    :return: a list of JD path
    '''
    data_dir = data_root + "JD2JL/"
    dir_lst = glob.glob(data_dir+"*")
    print(dir_lst)
    ret = []
    for item in dir_lst:
        ret = ret + glob.glob(item + "/*.txt")
    return ret
def get_JL_file():
    '''
    :return: a list of JL path
    '''
    data_dir = data_root + "JD2JL/"
    dir_lst = glob.glob(data_dir+"*")
    # print(dir_lst)
    tmp_lst = []
    ret = []
    for item in dir_lst:
        tmp_lst = tmp_lst + glob.glob(item + "/*.zip")
    for item in tmp_lst:
        print(item)
        zipf = zipfile.ZipFile(item)
        new_dir = item.replace(".zip","")
        zipf.extractall(new_dir)
        tmp = []
        tmp = tmp+glob.glob(new_dir + "/*")
        tmp_1 = []
        for it in tmp:
            tmp_1 = tmp_1+glob.glob(it+"/*")
        for it in tmp_1:
            ret = ret+glob.glob(it+"/*")
    return ret

def load_JD():
    result_dir = root_dir + "data/"
    path = result_dir + "JD.pkl"
    if os.path.exists(path):
        print("load JD from file")
        f = open(path,"rb")
        ret = pickle.load(f)
        f.close()
        return ret
    else:
        print("calculate and saving JD")
        JD_path_lst = get_JD_file()
        ret = []
        for it in JD_path_lst:
            ret.append(JD(it))
        f = open(path,"wb")
        ret[0].show_raw_data()
        pickle.dump(ret,f)
        f.close()
        return ret

def load_JL():
    result_dir = root_dir + "data/"
    path = result_dir + "JL.pkl"
    if os.path.exists(path):
        print("load JL from file")
        f = open(path,"rb")
        ret = pickle.load(f)
        f.close()
        return ret
    else:
        print("calculate and saving JL")
        JL_path_lst = get_JL_file()
        ret = []
        for it in JL_path_lst:
            ret.append(JL(it))
        f = open(path,"wb")
        ret[0].show_raw_data()
        pickle.dump(ret,f)
        f.close()
        return ret

# f = open(root_dir+"data/extracted_text.pkl","rb")
# sf = open(root_dir+"data/JDJL_corpus.txt","w",encoding="utf8")
#
# r = pickle.load(f)
# lst_1 = r["JD"]
# lst_2 = r["JL"]
# for line in lst_1+lst_2:
#     line = re.sub("\d、","",line)
#     line = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",line)
#     lst = jieba.cut(line)
#     str = " ".join(lst)
#     sf.write(str)
# sf.close()
# f.close()
# exit(0)


save_dic = {}
save_dic["JD"] = []
save_dic["JL"] = []
JD_lst = load_JD()
for tmp_JD in JD_lst:
    r = tmp_JD.extra_text_data()
    save_dic["JD"].append(r)
JL_lst = load_JL()
for tmp_JL in JL_lst:
    r = tmp_JL.extra_text_data()
    save_dic["JL"].append("".join(r))
f = open(root_dir+"/data/extracted_text.pkl","wb")
pickle.dump(save_dic,f)
f.close()
# for it in r:
#     print("@@@@@@@@@@@@@@@@")
#     print(it)
exit(0)

r = get_JL_file()
print(len(r))
print(r[:2])
jl = JL(r[0])
tmp_duty = jl.extra_text_data()
for it in tmp_duty:
    print("@@@@@@@@@@@@@@@@@@@@@")
    print(it)
exit(0)

r = get_JD_file()
jd = JD(r[0])
jd.show_raw_data()
