#coding = utf-8
import numpy as np
import sys, random, math
from collections import Counter
import os
import logging

np.random.seed(1)

# 加载数据，并返回按行读取结果
def loaddata(p):
    with open(p) as f:
        raws = f.readlines()  # 按行读取数据
    tokens = list(map(lambda x: x.split(" "), raws))  # 去除每条评论中的空格
    return tokens, raws

# 构建词汇表,将多行评论编码为数字list形式
def vocabulary(tokens):
    logging.debug("generate input dataset from comments")
    wordcnt = Counter()  # 计数器
    for sent in tokens:  # 每一条评论
        for word in sent:  # 评论中的每个单词
            wordcnt[word] -= 1  # 频率越高,值越小

    vocab = list(set(map(lambda x:x[0], wordcnt.most_common())))  # set防止重复,每个单词唯一一个编码
    word2index = {}
    for i, word in enumerate(vocab):  # 排序编码,比直接使用频率编码更紧凑
        word2index[word] = i
    return word2index

# 将评论tokens,按词汇表编码为input_dataset
def decode(word2index, tokens):
    input_dataset = list()  # 输入集
    concatenated = list()
    for sent in tokens:
        sent_indices = list()  # 编码每一条评论
        for word in sent:
            try:
                sent_indices.append(word2index[word])
                concatenated.append(word2index[word])  # 整个输入集的编码,不分评论的
            except:
                ""
        input_dataset.append(sent_indices)

    random.shuffle(input_dataset)  # 随机排列数字化评论
    concatenated = np.array(concatenated)
    return input_dataset, concatenated

# 逻辑回归函数
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# 训练神经网络
def train_network(input_dataset, word2index, concatenated):
    logging.debug("init parameter ...")
    alpha, iterations = (0.05, 2)               # 学习率, 迭代次数
    hidden_size, window, negative = (50, 2, 5)  # 隐藏层大小

    # weights_0_1 = (np.random.rand(len(vocab), hidden_size) - 0.5) * 0.2
    weights_0_1 = (np.random.rand(len(word2index), hidden_size) - 0.5) * 0.2
    weights_1_2 = np.random.rand(len(word2index), hidden_size)*0  # word2index 和 vocab 长度相同

    # 设置想要生成的标签
    layer_2_target = np.zeros(negative+1)
    layer_2_target[0] = 1  # [1, 0, 0, 0, 0, 0]

    for rev_i, review in enumerate(input_dataset):  # 每一条评论
        for target_i in range(len(review)):  # 每一个词汇
            target_samples = [review[target_i]] + \
            list(concatenated[(np.random.rand(negative)*len(concatenated)).astype('int').tolist()])
            # 某条评论中的一个单词 + 字典中随机的negative个单词
        
            left_context = review[max(0, target_i - window) : target_i]  # 指定单词左侧的window内的词序列
            right_context = review[target_i:min(len(review), target_i+window)]  # 指定单词右侧词序列

            # logging.debug("start train network ...")
            layer_1 = np.mean(weights_0_1[left_context + right_context], axis=0)  # 按列求平均值
            layer_2 = sigmoid(layer_1.dot(weights_1_2[target_samples].T))  # layer_1 to layer_2 的映射
            layer_2_delta = layer_2 - layer_2_target  # 直接增量
            layer_1_delta = layer_2_delta.dot(weights_1_2[target_samples])

            weights_0_1[left_context+right_context] -= layer_1_delta*alpha
            weights_1_2[target_samples] -= np.outer(layer_2_delta, layer_1)*alpha
    logging.debug("save weights_0_1 as file")
    np.save('weights', weights_0_1)
    return weights_0_1  # 网络训练的最终结果就是权重矩阵

# python 参数, 默认参数不能在普通参数前面, 默认参数的默认值需要定义在前
def similar_with_code(target, word2index, weights):
    logging.debug("find similar word with code from word2index")
    target_index = word2index[target]       # 查询目标词汇对应数字序列

    scores = Counter()   # 计数器,方便后面选用最相似的词
    for word, index in word2index.items():
        raw_difference = weights[target_index] - weights[index]  # 假设权重相似的词相似
        squared_difference = raw_difference*raw_difference  # 计算欧氏距离
        scores[word] = -math.sqrt(sum(squared_difference))
    return scores.most_common(10)   # 由大到小排列前十个

# 正则化参数
def normal(weights):
    norms = np.sum(weights*weights, axis=1)  # 正则化项
    norms.resize(norms.shape[0], 1)  # resize
    normed_weights = weights*norms  # 参数正则化, 点乘,对应位置相乘
    return normed_weights

# 类比,找到相似单词
# 相似度的前提假设是相似权重值对应相似的词汇
def analogy(weights, word2index, positive=['terrible', 'good'], negative=['bad']):
    logging.debug("find similar word from weights")
    normed_weights = normal(weights)  # 参数正则化

    query_vect = np.zeros(len(weights[0]))
    for word in positive:  # 词向量之间的加减
        query_vect += normed_weights[word2index[word]]
    for word in negative:
        query_vect -= normed_weights[word2index[word]]
    
    scores = Counter()
    for word, index in word2index.items():
        raw_difference = weights[index] - query_vect  # 词汇向量对应权重的差
        squared_difference = raw_difference*raw_difference
        scores[word] = -math.sqrt(sum(squared_difference))
    return scores.most_common(10)[1:]  # most_common 默认从大到小排列

# 将词序列转化为数字向量
def make_sent_vect(words, normed_weights):
    indices = list(map(lambda x: word2index[x], filter(lambda x:x in word2index, words)))
    # filter() 函数用于过滤序列,过滤掉不符合条件的元素,返回由符合条件元素组成的新列表
    return np.mean(normed_weights[indices], axis=0)
    # axis = i numpy 沿着第i个下标变化的方向进行操作

# 将评论集中的评论都转换为向量
def reviews2vectors(tokens, normed_weights):
    vectors = list()
    for review in tokens:
        vectors.append(make_sent_vect(review, normed_weights))  # 将一条评论转换为向量
    return np.array(vectors)  # 将list转化为数组array

# 
def most_similar_reviews(review, normed_weights, vectors, raw_reviews):
    vec = make_sent_vect(review, normed_weights)  # 将评论转化为向量
    scores = Counter()  # 计数器
    for i, val in enumerate(vectors.dot(vec)):  # 通过计算内积衡量距离
        scores[i] = val  # 统计频率
    most_similar = list()

    for idx, score in scores.most_common(3):  # 最相似的3条评论
        most_similar.append(raw_reviews[idx][0:40])  # 前40个词汇
    return most_similar

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s, %(filename)s, %(message)s")
    logging.debug("cloze starting ...")
    logging.debug("loading dataset ...")
    tokens, raw_reviews = loaddata("reviews.txt")
    logging.debug("generate vocabulary of comment ...")
    word2index = vocabulary(tokens)
    logging.debug("decode input dataset ...")
    input_dataset, concatenated = decode(word2index, tokens)
    if os.path.exists('weights.npy'):
        logging.debug("load weights without training")
        weights = np.load('weights.npy')
    else:
        logging.debug("training network ...")
        weights = train_network(input_dataset, word2index, concatenated)
    logging.debug(similar_with_code('terrible', word2index, weights))
    logging.debug(analogy(weights, word2index))
    logging.debug("decode words to normed weights ...")
    normed_weights = normal(weights)
    logging.debug("decode comment to vector ...")
    vectors = reviews2vectors(tokens, normed_weights)
    logging.debug(most_similar_reviews(['boring', 'awful'], normed_weights, vectors, raw_reviews))
    # print(make_sent_vect(['boring', 'awful'], normed_weights))

    print("there are many places that can be optimized ...")