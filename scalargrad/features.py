import random
from .core import Value

class MLP:
    def __init__(self,widths,seed=7):
        if len(widths)<2 or any(not isinstance(w,int) or w<1 for w in widths):raise ValueError('positive layer widths required')
        rng=random.Random(seed);self.layers=[]
        for ins,outs in zip(widths,widths[1:]):
            self.layers.append([([Value(rng.uniform(-1,1)) for _ in range(ins)],Value(0)) for _ in range(outs)])
    def parameters(self):return [p for layer in self.layers for weights,bias in layer for p in [*weights,bias]]
    def __call__(self,x):
        if len(x)!=len(self.layers[0][0][0]):raise ValueError('input width mismatch')
        for layer in self.layers:
            x=[(sum(wi*xi for wi,xi in zip(w,x))+b).tanh() for w,b in layer]
        return x

def fit(model,x,y,steps=1000,rate=.05):
    if not x or len(x)!=len(y) or steps<1 or rate<=0:raise ValueError('invalid training configuration')
    losses=[]
    for _ in range(steps):
        errors=[model(row)[0]-label for row,label in zip(x,y)]
        loss=sum(e*e for e in errors)*(1/len(x));loss.backward();losses.append(loss.data)
        for p in model.parameters():p.data-=rate*p.grad
    return losses
