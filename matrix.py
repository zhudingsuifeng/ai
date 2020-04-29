#coding = utf-8
import sys
import numpy as np
import logging
np.random.seed(1)

# 原始按行读取文件
def _loadlines(p):
    f = open(p)
    raw = f.readlines()
    f.close()
    return raw

# 更安全
def loadlines(p):
    with open(p) as f:
        raw = f.readlines()
    return raw
    
def sigmoid(x):
    return 1/(1 + np.exp(-x))

if __name__ == "__main__":
    # logging.basicConfig(filename='nlp.log', level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s, %(filename)s, %(message)s")  # 通过设置日志等级可以控制显示信息
    logging.debug("nlp is comming ...")

    logging.debug("load data ...")
    # f = open('reviews.txt')
    # raw_reviews = f.readlines()   # 每一行是一条评论
    # f.close()
    raw_reviews = loadlines('reviews.txt')

    # f = open('labels.txt')
    # raw_lables = f.readlines()    # 按行读取文件,返回list
    # f.close()
    raw_lables = loadlines('labels.txt')
    
    logging.debug("decode comments")
    tokens = list(map(lambda x: set(x.split(" ")), raw_reviews))  # 去除多余空格,split() 返回list
    vocab = set()
    for sent in tokens:
        for word in sent:
            if(len(word)>0):
                vocab.add(word)
    vocab = list(vocab)         # 所有评论的字典,所有评论共用同一个字典

    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word] = i    # 编码
    
    input_dataset = list()
    for sent in tokens:         # 每一条评论
        sent_indices = list()   # 每一条评论有文字转化为数字编码
        for word in sent:
            try:
                sent_indices.append(word2index[word])
            except:
                ""
        input_dataset.append(list(set(sent_indices)))   # 所有评论的数字编码
    
    logging.debug("decode score")
    target_dataset = list()
    for label in raw_lables:
        if label == 'positive\n':
            target_dataset.append(1)
        else:
            target_dataset.append(0)

    logging.debug("init learn rate and weights")
    alpha, iterations = (0.01, 2)   # 学习率，迭代次数
    hidden_size = 100               # 隐藏层大小

    weights_0_1 = 0.2*np.random.random((len(vocab), hidden_size)) - 0.1  # 权重矩阵
    weights_1_2 = 0.2*np.random.random((hidden_size, 1)) - 0.1  # 权重矩阵

    correct, total = (0, 0)
    logging.debug("train model ... ")
    for iter in range(iterations):
        for i in range(len(input_dataset)-1000):
            x, y = (input_dataset[i], target_dataset[i])       # 输入与输出
            logging.debug(weights_0_1[x])
            layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))  # x 向量,weights_0_1 矩阵中指定行按列求和
            layer_2 = sigmoid(np.dot(layer_1, weights_1_2))    # 矩阵乘法,激活函数

            layer_2_delta = layer_2 - y                        # 反向传播, error 影响力
            layer_1_delta = layer_2_delta.dot(weights_1_2.T)   # error 影响力反向传播到layer_1

            weights_0_1[x] -= layer_1_delta*alpha              # 更新权重
            weights_1_2 -= np.outer(layer_1, layer_2_delta)*alpha  # 
            logging.debug(weights_0_1[x])

    # logging.debug(vocab)
    print("hi, nlp")