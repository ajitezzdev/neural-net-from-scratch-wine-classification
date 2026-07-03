import random
import math


def Relu(value):
    out=Value(max(0,value.data),(value,))
    def _backward():
        if value.data>0:
            value.grad+=out.grad
    out.backward=_backward
    return out


class Value:

    def __init__(self,data,parents=()):
        self.data=data
        self.parents=list(parents)
        self.grad=0.0
        self.backward=lambda:None

    def __repr__(self):
        return f'data:{self.data} parents:{self.parents}'

    def __add__(self,other):
        out=Value(self.data+other.data,(self,other))
        def _backward():
            self.grad+=out.grad
            other.grad+=out.grad
        out.backward=_backward
        return out
    
    def __sub__(self,other):
        out=Value(self.data-other.data,(self,other))
        def _backward():
            self.grad+=out.grad
            other.grad-=out.grad
        out.backward=_backward
        return out

    def __mul__(self,other):
        out=Value(self.data*other.data,(self,other))
        def _backward():
            self.grad+=out.grad*other.data
            other.grad+=out.grad*self.data
        out.backward=_backward
        return out
    
    def __truediv__(self, other):
        out=Value(self.data/other.data,(self,other))
        def _backward():
            self.grad+=out.grad*(1/other.data)
            other.grad-=out.grad*(self.data/(other.data)**2)
        out.backward=_backward
        return out
    
    def __pow__(self, power):
        out = Value(self.data ** power, (self,))
        def _backward():
            self.grad+=out.grad * power * (self.data ** (power - 1))
        out.backward=_backward
        return out
    
    def __neg__(self):
        out = Value(-self.data, (self,))
        def _backward():
            self.grad -= out.grad
        out.backward =_backward
        return out
    
    def exp(self):
        out = Value(math.exp(self.data), (self,))
        def _backward():
            self.grad += out.grad * out.data
        out.backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data+1e-8), (self,))
        def _backward():
            self.grad += out.grad * (1/(self.data+1e-8))
        out.backward = _backward
        return out
    
    
class Neuron:

    def __init__(self,noip):
        self.w=[Value(random.uniform(-1,1)) for i in range(noip)]
        self.b=Value(random.uniform(-1,1))

    def __repr__(self):
        return f'w: {self.w} b:{self.b}'
    
    def forward(self,Xinp):
        out=Value(0)
        for w,x in zip(self.w,Xinp):
            out+=w*x
        out+=self.b
        return out
    
    def parameters(self):
        return self.w + [self.b]
    


class Layer:

    def __init__(self,noip,non,useRelu):  # Relu is not used in last layer
        self.neurons=[Neuron(noip) for i in range(non)]
        self.useRelu=useRelu

    def forward(self,Xinp):
        H=[]
        for neuron in self.neurons:
            out=neuron.forward(Xinp)
            if self.useRelu:
                pass
                out=Relu(out)
            H.append(out)
        return H
    
    def parameters(self):
        H=[]
        for neuron in self.neurons:
            H+=neuron.parameters()
        return H


class StdScaler:

    def __init__(self):
        self.mean=[]
        self.std=[]

    def fit(self,X):
        self.mean=[]
        self.std=[]
        for index in range(len(X[0])):
            Mean=0
            for sample in X:
                Mean+=sample[index]
            Mean=Mean/len(X)
            self.mean.append(Mean)

        for index in range(len(X[0])):
            stdDev=0
            for sample in X:
                stdDev+=(sample[index]-self.mean[index])**2
            stdDev=stdDev/len(X)
            stdDev=math.sqrt(stdDev)+1e-8
            self.std.append(stdDev)

    def transform(self,X):
        outerList=[]
        for sample in X:
            innerList=[]
            for val,mean,stddev in zip(sample,self.mean,self.std):
                out=(val-mean)/stddev
                innerList.append(out)
            outerList.append(innerList)
        return outerList
    
    def fit_transform(self,Xtrain):
        self.fit(Xtrain)
        return self.transform(Xtrain)


class Network:

    def __init__(self):
        self.layers=[
        Layer(13,16,True),
        Layer(16,16,True),
        Layer(16,8,True),
        Layer(8,3,False)]

    def forward(self,X):
        Xinp=[Value(i) for i in X]
        for layer in self.layers:
            Xinp=layer.forward(Xinp)
        return Xinp
    
    def parameters(self):
        H=[]
        for layer in self.layers:
            H+=layer.parameters()
        return H
    
    def get_state(self):
        state = []
        for layer in self.layers:
            layer_state = []
            for neuron in layer.neurons:
                neuron_state = {
                    "w": [w.data for w in neuron.w],
                    "b": neuron.b.data}
                
                layer_state.append(neuron_state)
            state.append(layer_state)
        return state

    def load_state(self, state):
        for layer, layer_state in zip(self.layers, state):
            for neuron, neuron_state in zip(layer.neurons, layer_state):
                for w, saved_w in zip(neuron.w, neuron_state["w"]):
                    w.data = saved_w
                neuron.b.data = neuron_state["b"]


    
def softmax(outLayer):
    H=[]
    total=Value(0)
    m = max(v.data for v in outLayer)
    maxVal=Value(m)
    expVal = [(v - maxVal).exp() for v in outLayer]
    for y in expVal:
        total += y
    for e in expVal:
        H.append(e / total)
    return H

def argmax(prob):
    Prob = [p.data for p in prob]
    return Prob.index(max(Prob))


def loss(X, Y, net):
    out = Value(0)
    for x, y in zip(X, Y):
        pred = softmax(net.forward(x))
        out += -pred[y].log()
    return out / Value(len(X))

def backprop(loss):
    topo=[]
    visited=set()
    def build(node):
        if node not in visited:
            visited.add(node)
            for parent in node.parents:
                build(parent)
            topo.append(node)
    build(loss)
    loss.grad=1
    for node in reversed(topo):
        node.backward()


def Train(net,X,Y,lr=0.01):
    Xloss=[]
    Yloss=[]
    for i in range(1000):
        Loss=loss(X,Y,net)
        backprop(Loss)
        for parameter in net.parameters():
            parameter.data-=lr*parameter.grad
            parameter.grad=0.0
        if i%10==0:
            print(i,Loss.data)
            Xloss.append(i)
            Yloss.append(Loss.data)
    return Xloss,Yloss








    

    
    
