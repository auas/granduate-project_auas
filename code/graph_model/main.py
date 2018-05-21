import pickle
import os
import glob
import numpy
import zipfile
import re
import pickle
import jieba
from pprint import pprint as pprint
import keras
from keras.layers import Input, Dense
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
root_dir = r"C:/Users/auas/Desktop/auas/大四/毕设/结题/code/key_word/"
data_root =  r"C:/Users/auas/Desktop/auas/大四/毕设/结题/data/"

def stat_matched_data(d):
    ret = []
    for id in d:
        ret.append(len(d[id]["JL"]))
    print(ret)

def load_data(p = data_root + "class_dataset.pkl"):
    f = open(p,"rb")
    x,y = pickle.load(f)
    x = np.array([it.reshape(128,) for it in x])
    print(x.shape)
    f.close()
    tr_num = int(0.8*len(y))
    tr = [x[:tr_num],y[:tr_num]]
    tst = [x[tr_num:],y[tr_num:]]
    return [tr,tst]

def build_class_model():
    input_arr = Input(shape=(128,))
    hid_1 = Dense(200, activation='relu')(input_arr)
    hid_2 = Dense(50, activation='relu')(hid_1)
    output =Dense(output_dim=1,activation="sigmoid")(hid_2)
    model = keras.models.Model(input_arr,output)

    sgd = keras.optimizers.RMSprop(lr=0.001)  # default 0.001

    model.compile(loss='binary_crossentropy', optimizer=sgd)
    return model



tr,tst = load_data()
auas_model = build_class_model()
checkpointer = keras.callbacks.ModelCheckpoint(
    filepath="classifer.checkpointer.hdf5",
    verbose=1, save_best_only=True)
earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=1)
print("training")
# auas_model.fit(tr[0],tr[1],epochs=1000,batch_size=20, shuffle=True, validation_split=0.1,
#                verbose=2,callbacks=[checkpointer,earlystopper])
print("loading model...")
auas_model.load_weights("classifer.checkpointer.hdf5")
print("predicting")
y_pred = auas_model.predict(tst[0])

def show_auc_result(y_pred,y):
    fpr, tpr, threshold = roc_curve(y, y_pred)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    plt.figure()
    lw = 2
    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for prediction')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    # plt.show()
print("showing result")
show_auc_result(y_pred,tst[1])
# auas_model.fit(tr_set)
# f = open(data_root+"sorted_data_final.pkl","rb")
# data = pickle.load(f)
# stat_matched_data(data)


























