#coding = utf-8

# writing like shakespeare
import sys,random,math
from collections import Counter
import numpy as np 
import logging
np.random.seed(1)

# 读取句子并返回小写去除换行和空格的句子list
def loaddata(p):
    with open(p) as f:
        raw = f.readlines()
    for line in raw[0:1000]:
        yield line.lower().replace("\n", "").split(" ")[1:]  # remove \n and black

# build the vocabulary
def vocabulary(tokens):
    vocab = set()
    for sent in tokens:
        for word in sent:
            vocab.add(word)  # 集合set添加元素

    return list(vocab)

# encode
def encode(tokens):
    vocab = vocabulary(tokens)
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word] = i
    return word2index

# decode
def words2indices(sentence, word2index):
    for word in sentence:
        yield word2index[word]  # 使用迭代器,在引用的时候需要list()

# softmax 激活函数
def softmax(x):
    e_x = np.exp(x - np.max(x))  # x is a vector,减去最大值是为了防止数值上溢
    return e_x/e_x.sum(axis=0)  # normalization, axis 沿第0个下标变化方向操作

# train model
def train(tokens):
    logging.debug("init parameter ... ")
    embed_size = 10  # 嵌入层大小
    word2index = encode(tokens)  # 获取编码表
    embed = (np.random.rand(len(word2index), embed_size) - 0.5) * 0.1  # encode嵌入权重矩阵,rand() 创建一个给定大小的矩阵，数据是随机填充的。
    recurrent = np.eye(embed_size)  # eye() 创建单位矩阵,共享权重矩阵
    start = np.zeros(embed_size)  # zeros() 创建零矩阵,空白句子
    decoder = (np.random.rand(embed_size, len(word2index)) - 0.5 ) * 0.1  # decode, 解码矩阵
    one_hot = np.eye(len(word2index))  # one hot decode

    logging.debug("start training model ...")
    for i in range(30000):  # 迭代
        alpha = 0.001  # learn rate
        sent = list(words2indices(tokens[i%len(tokens)][1:], word2index))  # 转码,向量化词汇,每一个元素都是单词向量
        layers = list()  # network layers
        layer = {}  # 每一层字典
        layer['hidden'] = start  # 输入单词
        layers.append(layer)  # 将中间层添加到网络中
        loss = 0  # 损失函数

        # logging.debug("前向传播, 预测下一个单词")
        for target_i in range(len(sent)):
            layer = {}
            layer['pred'] = softmax(layers[-1]['hidden'].dot(decoder))  # 最后一个隐藏层.dot编码权重矩阵
            loss += -np.log(layer['pred'][sent[target_i]])  # 矩阵[list], 负对数似然
            layer['hidden'] = layers[-1]['hidden'].dot(recurrent) + embed[sent[target_i]]
            layers.append(layer)

        # logging.debug("反向传播,更新权重")
        for layer_idx in reversed(range(len(layers))):  # 反向传播, reversed 返回一个反转的迭代器
            layer = layers[layer_idx]  # last hidden layer
            target = sent[layer_idx-1]  # 预测标签

            if layer_idx > 0:  # first layer and hidden layer
                layer['output_delta'] = layer['pred'] - one_hot[target]  # error
                new_hidden_delta = layer['output_delta'].dot(decoder.transpose())  # transpose() 不指定参数,默认矩阵转置, 反向传播
                if layer_idx == len(layers)-1:  # first layer
                    layer['hidden_delta'] = new_hidden_delta
                else:  # hidden layer
                    layer['hidden_delta'] = new_hidden_delta + layers[layer_idx+1]['hidden_delta'].dot(recurrent.transpose())
            else:
                layer['hidden_delta'] = layers[layer_idx+1]['hidden_delta'].dot(recurrent.transpose())  # last layer
            
        start -= layers[0]['hidden_delta']*alpha/float(len(sent))
        for layer_idx, layer in enumerate(layers[1:]):
            decoder -= np.outer(layers[layer_idx]['hidden'], layer['output_delta'])*alpha/float(len(sent))
            embed_idx = sent[layer_idx]
            embed[embed_idx] -= layers[layer_idx]['hidden']*alpha/float(len(sent))
            recurrent -= np.outer(layers[layer_idx]['hidden'], layer['hidden'])*alpha/float(len(sent))

        if i%1000 == 0:
            print("perplexity:"+str(np.exp(loss/len(sent))))
    return embed, recurrent, decoder

# 根据输入的前置单词,预测下一个单词
def predict(sent, embed, recurrent, decoder):
    embed_size = 10
    layers = list()  # network layers
    layer = {}  # 每一层字典
    layer['hidden'] = np.zeros(embed_size)  # 输入单词
    layers.append(layer)  # 将中间层添加到网络中
    
    preds = list()  # 预测结果
    for target_i in range(len(sent)):
        layer = {}
        layer['pred'] = softmax(layers[-1]['hidden'].dot(decoder))  # 最后一个隐藏层.dot编码权重矩阵
        layer['hidden'] = layers[-1]['hidden'].dot(recurrent) + embed[sent[target_i]]
        layers.append(layer)
    return layers

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s,%(filename)s,%(message)s")
    logging.debug("writing a novel like shakespeare ...")
    tokens = list(loaddata("qa1_single-supporting-fact_train.txt"))
    word2index = encode(tokens)  # 编码词汇为向量
    logging.debug("train model ...")
    embed, recurrent, decoder = train(tokens)  # 嵌入权重,循环权重,解码器权重
    logging.debug("predict the next word")
    sent_index = 4
    logging.debug(tokens[sent_index])
    layers = predict(list(words2indices(tokens[sent_index], word2index)), embed, recurrent, decoder)
    vocab = vocabulary(tokens)  # 词汇表
    for i, each_layer in enumerate(layers[1:-1]):
        inp = tokens[sent_index][i]
        true = tokens[sent_index][i+1]
        pred = vocab[each_layer['pred'].argmax()]
        print("prev input:" + inp + (' ' * (12 - len(inp))) + "true:" + true + (" " * (15 - len(true))) + "pred:" + pred)

    print("To be or not to be, that is the question.")