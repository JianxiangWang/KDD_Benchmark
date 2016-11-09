#!/usr/bin/env python
#encoding: utf-8
import socket
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# 当前工作目录
CWD = "/home/jianxiang/pycharmSpace/KDD_benchmark"


DATA_PATH = CWD + "/data"
DATASET_PATH = DATA_PATH + "/dataset"

# 训练和测试文件
TRAIN_FILE = DATASET_PATH + "/train_set/Train.csv"
TEST_FILE = DATASET_PATH + "/valid_set/Valid.csv"

# 模型文件
MODEL_PATH = CWD + "/model/kdd.model"
# 训练和测试特征文件
TRAIN_FEATURE_PATH = CWD + "/feature/train.feature"
TEST_FEATURE_PATH = CWD + "/feature/test.feature"
# 分类在测试级上的预测结果
TEST_RESULT_PATH = CWD + "/predict/test.result"
# 重新格式化的预测结果
TEST_PREDICT_PATH = CWD + "/predict/test.predict"



COAUTHOR_FILE = DATASET_PATH + "/coauthor.json"
PAPERIDAUTHORID_TO_NAME_AND_AFFILIATION_FILE = DATASET_PATH + "/paperIdAuthorId_to_name_and_affiliation.json"
PAPERAUTHOR_FILE = DATASET_PATH + "/PaperAuthor.csv"
AUTHOR_FILE = DATASET_PATH + "/Author.csv"

