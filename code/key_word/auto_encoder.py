import keras
from keras.layers import Input, Dense
import pickle
import numpy as np

root_dir = r"C:/Users/auas/Desktop/auas/大四/毕设/结题/code/key_word/"

def load_data(path):
    f = open(path,"rb")
    d = pickle.load(f)
    return d
def build_encoder():
    input_arr = Input(shape=(2055,))
    encoded = Dense(500, activation='relu')(input_arr)
    encoded = Dense(250, activation='relu')(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(250, activation='relu')(encoded)
    decoded = Dense(500, activation='relu')(decoded)
    decoded = Dense(2055, activation='sigmoid')(decoded)
    model = keras.models.Model(input_arr,decoded)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy')
    test_model = keras.models.Model(input_arr,encoded)
    return model,test_model

data = load_data(root_dir+"data/onehot_vec_matched_final.pkl")
JL_vec = data["JL"]
y = data["y"]
def PCA(X,y):
    from sklearn.decomposition import PCA
    import numpy as np
    import matplotlib.pyplot as plt
    pca = PCA(n_components=3)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    X_new = pca.transform(X)
    # print(X.shape)
    # print(np.array(X_new).shape)
    # exit(111)
    data_idx_lst = list(set(y))

    for data_idx in data_idx_lst:
        plt.scatter(X_new[y==data_idx, 0], X_new[y==data_idx, 1],label=str(data_idx),
                    color=1./data_idx)
    plt.legend(loc = "best")
    plt.show()
# PCA(np.concatenate([tr,tst]))
# exit(0)
tr_model,test_model_ = build_encoder()
# checkpointer = keras.callbacks.ModelCheckpoint(
#     filepath="encoder.checkpointer.hdf5",
#     verbose=1, save_best_only=True)
# earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=1)
print("training")
# tr_model.fit(tr,tr,epochs=1000,batch_size=100, shuffle=True, validation_split=0.1,
#                verbose=2,callbacks=[checkpointer,earlystopper])
print("testing")
tr_model.load_weights("encoder.checkpointer.hdf5")
encoder = tr_model.layers[3].output
input_arr = tr_model.input
test_model = keras.models.Model(input_arr,encoder)
pred = test_model.predict(JL_vec)

PCA(pred,y)
print(r[0])
