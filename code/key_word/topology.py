'''
abstract class for key word extract class: CRF and LSTM_CRF
'''
import json
class key_word_extract_model(object):
    def __init__(self,model_type):
        print("init_topology for:" + model_type)
        return
    def show_result(self,y_pred,y):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for y1,y2 in zip(y,y_pred):
            for yy1,yy2 in zip(y1,y2):
                if yy1 == "1":
                    if yy2 == "1":
                        TP+=1
                    else:
                        FP+=1
                if yy1 == "0":
                    if yy2 == "0":
                        TN+=1
                    else:
                        FN+=1

        prec = TP / (TP + FP)
        recall = TP/(TP+FN)
        F1 = 2 * TP / (2 * TP + FP + FN)

        print("TP:{0} TN:{1}".format(TP,TN))
        print("FP:{0} FN:{1}".format(FP,FN))
        print("prec: " + str(prec))
        print("recall: " + str(recall))
        print("F1: " + str(F1))
        return [prec,recall,F1]
    def load_raw_data(self,data_path):
        '''
        give a data path return the training and test data for crf
        :param data_path:
        :return: a list, each item is a dict {"tag","word"}
        '''
        f = open(data_path, "r", encoding="utf8")
        raw_data = json.load(f)
        tr_num = int(0.8 * len(raw_data))
        self.training_set = raw_data[:tr_num]
        self.test_set = raw_data[tr_num:]
        return
    def load_model(self,model_path):
        self.model_path = model_path
        self.model = None
        # print("in father load model: ",self.model)
        return
    def test_model(self,data_path,model_path):
        self.load_training_data(data_path)
        self.load_model(model_path)
        # print("father: ",self.model)
        if self.model == None:
            print("model is not defined yet!")
        return