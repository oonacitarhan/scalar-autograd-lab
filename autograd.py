"""Educational scalar autodiff implementation; standard reverse-mode chain rule."""
import math

class Value:
    def __init__(self, data, parents=(), backward=lambda:None):
        self.data, self.grad = float(data), 0.0
        self.parents, self._backward = parents, backward
    def __add__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data+other.data,(self,other))
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out
    __radd__ = __add__
    def __mul__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data*other.data,(self,other))
        def backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backward = backward
        return out
    __rmul__ = __mul__
    def __neg__(self): return self*-1
    def __sub__(self, other): return self+-other
    def tanh(self):
        out = Value(math.tanh(self.data),(self,))
        def backward(): self.grad += (1-out.data**2)*out.grad
        out._backward = backward
        return out
    def backward(self):
        order, seen = [], set()
        def visit(node):
            if node not in seen:
                seen.add(node)
                for parent in node.parents: visit(parent)
                order.append(node)
        visit(self)
        for node in order: node.grad = 0
        self.grad = 1
        for node in reversed(order): node._backward()

def train(steps=200):
    w,b = Value(0),Value(0)
    data = [(-1,-1),(0,1),(1,3),(2,5)]
    losses=[]
    for _ in range(steps):
        errors=[w*x+b-y for x,y in data]
        loss=sum(e*e for e in errors)*.25
        loss.backward()
        losses.append(loss.data)
        for p in (w,b): p.data -= .05*p.grad
    return w.data,b.data,losses

def self_test():
    x=Value(.7)
    result=(x*x+x).tanh()
    result.backward()
    f=lambda t:math.tanh(t*t+t)
    numeric=(f(.700001)-f(.699999))/.000002
    assert abs(x.grad-numeric)<1e-6
    a=Value(3); (a*a+a).backward(); assert a.grad==7
    w,b,loss=train()
    assert loss[-1]<loss[0]*1e-5
    assert abs(w-2)<.01 and abs(b-1)<.01
    return w,b,loss[-1]

if __name__=='__main__':
    print('Learned w, b, final loss:', self_test())
    print('Self-tests passed')
