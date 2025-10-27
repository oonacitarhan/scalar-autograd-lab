from scalargrad.core import *

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

def test_original_reference_cases():
    self_test()
