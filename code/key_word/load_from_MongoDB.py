from pymongo import MongoClient
import pdb
from pprint import pprint
import pickle
# connect to the MongoDB on MongoLab
# to learn more about MongoLab visit http://www.mongolab.com
# replace the "" in the line below with your MongoLab connection string
# you can also use a local MongoDB instance

class read_MongoDB():
    def __init__(self):
        return
    def extra_txt(self,pre_dic):
        raise NotImplemented
    def save_pkl(self,save_path,collection):
        self.f = open(save_path,"wb")
        self.col = collection.find()
        ret = []
        count = 0
        for idx,item in enumerate(self.col):
            tmp_txt = self.extra_txt(item)
            ret.append(tmp_txt)
            count = idx
        print("there are {0} collections".format(count))
        pickle.dump(ret,self.f)
        self.f.close()

class read_51job_cv(read_MongoDB):
    def __init__(self):
        super(read_51job_cv).__init__()
        return
    def extra_txt(self,pre_dic):
        ret = ""
        work_lst = pre_dic["work"]
        project_lst = pre_dic["project"]
        ret = ret+pre_dic["self_evaluation"] + " "
        for work in work_lst:
            ret = ret + " " + work["description"]
        for prj in project_lst:
            ret = ret + " " + prj["duty"] + " " + prj["description"]
        return ret.replace("\n"," ")


def read_25job(collenction):
    ret = []
    try:
        for item in collenction.find():
            ret.append(read_)
    except:
        print("get_wrong")

JL_url = "mongodb://root:bubb100178@106.14.147.212:3717"
JD_url = "mongodb://root:Pinnacle20182018@101.132.42.226:5238"

# connStr = "mongodb://root:buaa100191@106.14.147.212:3718/admin"
JL_c = MongoClient(JL_url)
db = JL_c.cv
coll_51_job = db["51job"]
auas_loader_51job  = read_51job_cv()
save_path = "51job.pkl"
auas_loader_51job.save_pkl(save_path,coll_51_job)