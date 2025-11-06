from scalargrad.features import MLP,fit
def test_xor_training():
    x=[[-1,-1],[-1,1],[1,-1],[1,1]];y=[-1,1,1,-1]
    model=MLP([2,4,1]);losses=fit(model,x,y,1000)
    assert losses[-1]<.04
    assert all(model(row)[0].data*label>0 for row,label in zip(x,y))
def test_parameter_gradient():
    model=MLP([2,2,1]);p=model.parameters()[0];original=p.data
    out=model([.2,.7])[0];out.backward();analytic=p.grad
    p.data=original+1e-6;right=model([.2,.7])[0].data
    p.data=original-1e-6;left=model([.2,.7])[0].data;p.data=original
    assert abs(analytic-(right-left)/2e-6)<1e-6
def test_shapes():
    assert len(MLP([2,3,1]).parameters())==13
def test_bad_input():
    try:MLP([2,1])([0])
    except ValueError:return
    raise AssertionError('bad width accepted')
