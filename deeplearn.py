#coding = utf-8
import numpy as np
from collections import Counter
import sys, random, math
import logging

np.random.seed(0)

class Tensor(object):
    
    def __init__(self, data, autograd=False, creators=None, creation_op=None, mid=None):
        self.data = np.array(data)  # 自身数据
        self.creation_op = creation_op  # 操作
        self.creators = creators  # 操作数
        self.grad = None  # 梯度
        self.autograd = autograd  # 自动求梯度
        self.children = {}  # 孩子字典
        if mid is None:
            mid = np.random.randint(0, 100000)  # 随机生成一个节点id
        self.mid = mid  # 节点id
        if creators is not None:
            for c in creators:  # 操作数,其实就是父节点
                if self.mid not in c.children:
                    c.children[self.mid] = 1  # 父节点梯度传播首次经过当前节点
                else:
                    c.children[self.mid] += 1  # 梯度传播经过当前节点次数加一

    # 计算一个张量是否已经从它在计算图中的所有孩子那里接收到了梯度
    def all_children_grads_accounted_for(self):
        for mid, cnt in self.children.items():
            if cnt != 0:
                return False
        return True

    # 反向传播
    def backward(self, grad=None, grad_origin=None):
        if self.autograd:  # 自动传播
            if grad is None:
                grad = Tensor(np.ones_like(self.data))
            if grad_origin is not None:
                if self.children[grad_origin.mid] == 0:  # 子节点已经反向传播完毕
                    return   # 后面添加的,不然一直报错
                    raise Exception("cannot backprop more than once")
                else:
                    self.children[grad_origin.mid] -= 1  # 反向传播子节点数量减一
            if self.grad is None:
                self.grad = grad  # 首次求梯度
            else:
                self.grad += grad  # 梯度累加

            assert grad.autograd == False

            if self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is None):
                # 子节点梯度接收完毕,开始下一层的反向传播
                # 加法反向传播
                if self.creation_op == "add":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
                # 取负值反向传播
                if self.creation_op == "neg":
                    self.creators[0].backward(self.grad.__neg__())
                # 减法反向传播,减法相当于加法和取负值的组合
                if self.creation_op == "sub":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad.__neg__(), self)
                # 乘法反向传播
                if self.creation_op == "mul":
                    self.creators[0].backward(self.grad*self.creators[1], self)
                    self.creators[1].backward(self.grad*self.creators[0], self)
                # 矩阵乘法反向传播
                if self.creation_op == "mm":
                    act = self.creators[0]  # 通常是激活函数
                    weights = self.creators[1]  # 通常是权重矩阵
                    new = self.grad.mm(weights.transpose())
                    act.backward(new)
                    new = self.grad.transpose().mm(act).transpose()
                    weights.backward(new)
                # 转置反向传播
                if self.creation_op == "transpose":
                    self.creators[0].backward(self.grad.transpose())
                # 指定维度求和反向传播
                if "sum" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])  # 需要求和的变化坐标
                    ds = self.creators[0].data.shape[dim]  # 要求和维度元素数量
                    self.creators[0].backward(self.grad.expand(dim, ds))
                # 扩展运算反向传播
                if "expand" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))  # sum 和 expand互为逆运算
                if self.creation_op == "sigmoid":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad*(self*(ones-self)))
                if self.creation_op == "tanh":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad*(ones-(self*self)))
                if self.creation_op == "index_select":
                    new_grad = np.zeros_like(self.creators[0].data)  # 零矩阵
                    indices_ = self.index_select_indices.data.flatten()  # 平铺
                    grad_ = grad.data.reshape(len(indices_), -1)
                    for i in range(len(indices_)):
                        new_grad[indices_[i]] += grad_[i]  # 反向传播,更新对应位置梯度
                    self.creators[0].backward(Tensor(new_grad))
                if self.creation_op == "cross_entropy":
                    dx = self.softmax_output - self.target_dist
                    self.creators[0].backward(Tensor(dx))

    # 加法运算
    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, autograd=True, creators=[self, other], creation_op="add")
        return Tensor(self.data + other.data)

    # 取负值运算
    def __neg__(self):
        if self.autograd:
            return Tensor(self.data*-1, autograd=True, creators=[self], creation_op="neg")
        return Tensor(self.data*-1)

    # 减法运算
    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data-other.data, autograd=True, creators=[self, other], creation_op="sub")
        return Tensor(self.data-other.data)

    # 乘法运算
    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data*other.data, autograd=True, creators=[self, other], creation_op="mul")
        return Tensor(self.data*other.data)

    # 沿指定dim坐标变化方向求和运算
    def sum(self, dim):  # dim 沿着坐标变化的方向求和
        if self.autograd:
            return Tensor(self.data.sum(dim), autograd=True, creators=[self], creation_op="sum_"+str(dim))
        return Tensor(self.data.sum(dim))

    # 扩充运算,sum的逆运算 dim 就像剥洋葱,[0, 1, 2] 越靠后的越是内层
    def expand(self, dim, copies):  # dim扩展维度,copies扩展次数
        trans_cmd = list(range(0, len(self.data.shape)))  # 维度
        trans_cmd.insert(dim, len(self.data.shape))  #insert(d, i) 在索引位置i插入数值d 
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)  # 转置transpose(0,1,2) 保持不变,transpose(1,0,2) 0轴和1轴互换
        if self.autograd:
            return Tensor(new_data, autograd=True, creators=[self], creation_op="expand_"+str(dim))
        return Tensor(new_data)

    # 转置运算
    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(), autograd=True, creators=[self], creation_op="transpose")
        return Tensor(self.data.transpose())

    # 矩阵乘法运算
    def mm(self, x):
        if self.autograd:
            return Tensor(self.data.dot(x.data), autograd=True, creators=[self, x], creation_op="mm")
        return Tensor(self.data.dot(x.data))
    
    # 下标操作
    def index_select(self, indices):
        if self.autograd:
            new = Tensor(self.data[indices.data], autograd=True, creators=[self], creation_op="index_select")
            new.index_select_indices = indices  # 按行选取的向量list
            return new
        return Tensor(self.data[indices.data])

    def sigmoid(self):
        if self.autograd:
            return Tensor(1/(1+np.exp(-self.data)), autograd=True, creators=[self], creation_op="sigmoid")
        return Tensor(1/(1+np.exp(-self.data)))

    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data), autograd=True, creators=[self], creation_op="tanh")
        return Tensor(np.tanh(self.data))

    # 交叉熵
    def cross_entropy(self, target_indices):
        temp = np.exp(self.data)
        softmax_output = temp/np.sum(temp, axis=len(self.data.shape)-1, keepdims=True)
        t = target_indices.data.flatten()
        p = softmax_output.reshape(len(t), -1)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p)*(target_dist)).sum(1).mean()
        if self.autograd:
            out = Tensor(loss, autograd=True, creators=[self], creation_op="cross_entropy")
            out.softmax_output = softmax_output
            out.target_dist = target_dist
            return out
        return Tensor(loss)


    def __repr__(self):
        return str(self.data.__repr__())  # 面向开发人员,展示类的成员属性

    def __str__(self):
        return str(self.data.__str__())  # 把一个类的实例变成str,执行print()时，输出的内容

# 梯度下降
class SGD(object):
    def __init__(self, parameters, alpha=0.1):
        self.parameters = parameters  # 参数,权重
        self.alpha = alpha  # 学习率

    def zero(self):
        for p in self.parameters:  # 所有权重
            p.grad.data *= 0

    def step(self, zero=True):
        for p in self.parameters:
            p.data -= p.grad.data * self.alpha  # 更新变量
            if zero:
                p.grad.data *= 0

# 抽象层
class Layer(object):
    def __init__(self):
        self.parameters = list()

    def get_parameters(self):
        return self.parameters

# 线性层
class Linear(Layer):
    def __init__(self, n_inputs, n_outputs, bias=True):
        super().__init__()
        W = np.random.randn(n_inputs, n_outputs)*np.sqrt(2.0/(n_inputs))  # 初始化权重
        self.weight = Tensor(W, autograd=True)  # 权重张量
        self.bias = Tensor(np.zeros(n_outputs), autograd=True)  # 偏置
        self.parameters.append(self.weight)  # 参数
        self.parameters.append(self.bias)

    def forward(self, inp):
        return inp.mm(self.weight) + self.bias.expand(0, len(inp.data))

# 序贯模型
class Sequential(Layer):
    def __init__(self, layers=list()):
        super().__init__()
        self.layers = layers

    # 添加层
    def add(self, layer):
        self.layers.append(layer)

    # 前向传播
    def forward(self, inp):
        for l in self.layers:
            inp = l.forward(inp)
        return inp

    #
    def get_parameters(self):
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params

class MSELoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return ((pred-target)*(pred-target)).sum(0)

class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return inp.tanh()

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return inp.sigmoid()

class Embedding(Layer):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        # weight = (np.random.rand(vocab_size, dim)-0.5)/dim
        self.weight = Tensor((np.random.rand(vocab_size, dim)-0.5)/dim, autograd=True)
        self.parameters.append(self.weight)

    def forward(self, inp):
        return self.weight.index_select(inp)

class CrossEntropyLoss(object):
    def __init__(self):
        super().__init__()

    def forward(self, inp, target):
        return inp.cross_entropy(target)

class RNNCell(Layer):
    def __init__(self, n_inputs, n_hidden, n_output, activation="sigmoid"):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output
        if activation == "sigmoid":
            self.activation = Sigmoid()
        elif activation == "tanh":
            self.activation = Tanh()
        else:
            raise Exception("Non-linearity not found")

        self.w_ih = Linear(n_inputs, n_hidden)
        self.w_hh = Linear(n_hidden, n_hidden)
        self.w_ho = Linear(n_hidden, n_output)

        self.parameters += self.w_ih.get_parameters()
        self.parameters += self.w_hh.get_parameters()
        self.parameters += self.w_ho.get_parameters()

    def forward(self, inp, hidden):
        from_prev_hidden = self.w_hh.forward(hidden)  # 隐藏层
        combined = self.w_ih.forward(inp) + from_prev_hidden  # 输入层和隐藏层
        new_hidden = self.activation.forward(combined)  # 激活
        output = self.w_ho.forward(new_hidden)  # 输出层
        return output, new_hidden

    def init_hidden(self, batch_size=1):
        return Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)

class LSTMCell(Layer):
    def __init__(self, n_inputs, n_hidden, n_output):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.xf = Linear(n_inputs, n_hidden)  # 遗忘
        self.xi = Linear(n_inputs, n_hidden)  # 输入
        self.xo = Linear(n_inputs, n_hidden)  # 输出
        self.xc = Linear(n_inputs, n_hidden)  # 隐层状态
        self.hf = Linear(n_hidden, n_hidden, bias=False)
        self.hi = Linear(n_hidden, n_hidden, bias=False)
        self.ho = Linear(n_hidden, n_hidden, bias=False)
        self.hc = Linear(n_hidden, n_hidden, bias=False)

        self.w_ho = Linear(n_hidden, n_output, bias=False)

        self.parameters += self.xf.get_parameters()
        self.parameters += self.xi.get_parameters()
        self.parameters += self.xo.get_parameters()
        self.parameters += self.xc.get_parameters()
        self.parameters += self.hf.get_parameters()
        self.parameters += self.hi.get_parameters()
        self.parameters += self.ho.get_parameters()
        self.parameters += self.hc.get_parameters()

        self.parameters += self.w_ho.get_parameters()

    def forward(self, inp, hidden):
        prev_hidden = hidden[0]
        prev_cell = hidden[1]
        f = (self.xf.forward(inp)+self.hf.forward(prev_hidden)).sigmoid()
        i = (self.xi.forward(inp)+self.hi.forward(prev_hidden)).sigmoid()
        o = (self.xo.forward(inp)+self.ho.forward(prev_hidden)).sigmoid()
        g = (self.xc.forward(inp)+self.hc.forward(prev_hidden)).tanh()
        c = (f*prev_cell)+(i*g)
        h = o*c.tanh()
        output = self.w_ho.forward(h)
        return output, (h, c)

    def init_hidden(self, batch_size=1):
        h = Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)
        c = Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)
        h.data[:,0] += 1
        c.data[:,0] += 1
        return (h, c)

# 训练网络
def train(p, iterations=400):
    with open(p, 'r') as f:
        raw = f.read()

    vocab =list(set(raw))
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word] = i  # 词汇映射表
    indices = np.array(list(map(lambda x: word2index[x], raw)))  # encode
    embed = Embedding(vocab_size=len(vocab), dim=512)
    model = LSTMCell(n_inputs=512, n_hidden=512, n_output=len(vocab))
    model.w_ho.weight.data *= 0
    critersion = CrossEntropyLoss()
    optim = SGD(parameters=model.get_parameters() + embed.get_parameters(), alpha=0.05)
    batch_size = 16
    bptt = 25
    n_batches = int((indices.shape[0]/(batch_size)))

    trimmed_indices = indices[:n_batches*batch_size]
    batched_indices = trimmed_indices.reshape(batch_size, n_batches)
    batched_indices = batched_indices.transpose()
    input_batched_indices = batched_indices[0:-1]
    target_batched_indices = batched_indices[1:]

    n_bptt = int((n_batches-1)/bptt)
    input_batches = input_batched_indices[:n_bptt*bptt]
    input_batches = input_batches.reshape(n_bptt, bptt, batch_size)
    target_batches = target_batched_indices[:n_bptt*bptt]
    target_batches = target_batches.reshape(n_bptt, bptt, batch_size)
    min_loss = 1000

    for i in range(iterations):
        logging.debug("on " + str(i) + "'s training ...")
        total_loss, n_loss = (0, 0)
        hidden = model.init_hidden(batch_size=batch_size)
        batches_to_train = len(input_batches)

        for batch_i in range(batches_to_train):
            if batch_i < 3:
                logging.debug("point one ...")
            hidden = (Tensor(hidden[0].data, autograd=True), Tensor(hidden[1].data, autograd=True))
            losses = list()

            for t in range(bptt):
                if t == 0:
                    logging.debug("point two ...")
                inp = Tensor(input_batches[batch_i][t], autograd=True)
                rnn_input = embed.forward(inp=inp)
                output, hidden = model.forward(inp=rnn_input, hidden=hidden)
                target = Tensor(target_batches[batch_i][t], autograd=True)
                batch_loss = critersion.forward(output, target)
                if t == 0:
                    losses.append(batch_loss)
                else:
                    losses.append(batch_loss+losses[-1])
            
            loss = losses[-1]
            loss.backward()
            optim.step()
            total_loss += loss.data/bptt
            epoch_loss = np.exp(total_loss/(batch_i+1))
            if epoch_loss < min_loss:
                min_loss = epoch_loss
        optim.alpha *= 0.99
    return model, word2index, vocab, embed

def generate_sample(model, word2index, vocab, embed, n=30, init_char=' '):
    s = ""
    hidden = model.init_hidden(batch_size=1)
    inp = Tensor(np.array([word2index[init_char]]))
    for i in range(n):
        rnn_input = embed.forward(inp)
        output, hidden = model.forward(inp=rnn_input, hidden=hidden)
        output.data *= 15
        temp_dist = output.softmax()
        temp_dist /= temp_dist.sum()

        m = output.data.argmax()
        c = vocab[m]
        inp = Tensor(np.array([m]))
        s += c
    return s

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s, %(filename)s, %(message)s")  # 通过设置日志等级可以控制显示信息
    p = "shakespear.txt"
    logging.debug("start training the model ...")
    model, word2index, vocab, embed = train(p, 6)
    logging.debug("generate the sample ...")
    print(generate_sample(model, word2index, vocab, embed, n=500, init_char='\n'))

    logging.debug("build my own deep learn frame ...")