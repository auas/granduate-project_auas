# -*- coding:utf-8 -*-
import sys, pickle, os, random
import numpy as np
import io

## tags, BIO
pre_tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }
tag2label = {u'0': 0,u'1': 1}


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with io.open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        # print("auas    ",line)
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data
# root = r"C:\Users\auas\Desktop\auas\大四\毕设\中期"
root = "/home/jinsh/wiki_model/data/"
def auas_read_corpus(corpus_path = "extraction.corpus_all.json"):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    json_data = io.open(corpus_path, "r", encoding="utf8")
    # r = json.load(json_data)
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
            # print(tmp_dic)
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
    for tmp_dic in rr4[1:]:
        data.append((tmp_dic["word"], tmp_dic["tag"]))
    tot_len = len(data)
    tr = data[:int(0.8*tot_len)]
    tst = data[int(0.8*tot_len):]
    return [tr,tst]
def vocab_build(vocab_path, corpus_path = root + r"\extraction.corpus_all.json", min_count=1):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    tr,tst = auas_read_corpus(corpus_path)
    data = tr+tst
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            # elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
            #     word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with io.open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw,protocol=2)
# auas_data_root = "/home/jinsh/wiki_model/data/"
# auas_data_path = r"C:\Users\auas\Desktop\auas\大四\毕设\中期之后\new_word2id.pkl"
# vocab_build(auas_data_path)
# exit(0)


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with io.open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

