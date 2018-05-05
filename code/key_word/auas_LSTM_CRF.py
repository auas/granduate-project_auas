# -*- coding:utf-8 -*-
import os
import sys
root_dir = "/home/jinsh/wiki_model/code/"
word2id_path = root_dir + "key_word/best_model/new_word2id.pkl"
sys.path.append(root_dir+"key_word/best_model/checkpoints/")
sys.path.append(root_dir+"biLSTM_CRF/")
## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import numpy as np
import os, argparse, time, random
from model import BiLSTM_CRF
from utils import str2bool, get_logger, get_entity
from data import auas_read_corpus, read_dictionary, tag2label, random_embedding

"1525413493"

def para_set():
    parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
    parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
    parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
    parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
    parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
    parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
    parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
    parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
    parser.add_argument('--pretrain_embedding', type=str, default='random',
                        help='use pretrained char embedding or init it randomly')
    parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
    parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
    parser.add_argument('--mode', type=str, default='test', help='train/test/demo')
    parser.add_argument('--demo_model', type=str, default='1522161339', help='model for test and demo')
    args = parser.parse_args()
    return args

def path_set():
    ## paths setting
    paths = {}
    timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
    output_path = os.path.join('.', args.train_data + "_save", timestamp)
    if not os.path.exists(output_path): os.makedirs(output_path)
    summary_path = os.path.join(output_path, "summaries")
    paths['summary_path'] = summary_path
    if not os.path.exists(summary_path): os.makedirs(summary_path)
    model_path = os.path.join(output_path, "checkpoints/")
    if not os.path.exists(model_path): os.makedirs(model_path)
    ckpt_prefix = os.path.join(model_path, "model")
    paths['model_path'] = ckpt_prefix
    result_path = os.path.join(output_path, "results")
    paths['result_path'] = result_path
    if not os.path.exists(result_path): os.makedirs(result_path)
    log_path = os.path.join(result_path, "log.txt")
    paths['log_path'] = log_path
    get_logger(log_path).info(str(args))
    return paths

args = para_set()
paths = path_set()
config = tf.ConfigProto()
print("loading data")
train_data, test_data = auas_read_corpus("/home/jinsh/wiki_model/data/extraction.corpus_all.json")
print("{0} training data \n{1} test data".format(len(train_data), len(test_data)))
# always use random embedding

word2id = read_dictionary(word2id_path)
embeddings = random_embedding(word2id, args.embedding_dim)
model_path = root_dir +"key_word/best_model/checkpoints/"

ckpt_file = tf.train.latest_checkpoint(model_path)
# print(ckpt_file)
print(ckpt_file)
# exit
paths['model_path'] = ckpt_file
model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
model.build_graph()
# print("test data: {}".format(len(test_data)))
model.test(test_data)