import subprocess,sys
def test_help():
    result=subprocess.run([sys.executable,'-m','scalargrad','--help'],text=True,capture_output=True)
    assert result.returncode==0 and 'usage:' in result.stdout
def test_cli_rejects_unknown_argument():
    result=subprocess.run([sys.executable,'-m','scalargrad','--unknown-option'],text=True,capture_output=True)
    assert result.returncode!=0

def test_documented_workflow():
    import json
    result=subprocess.run([sys.executable,'-m','scalargrad',*['--steps','1000','--seed','0']],text=True,capture_output=True)
    assert result.returncode==0, result.stderr
    data=json.loads(result.stdout)
    assert data['final_loss']<.04 and all(a*b>0 for a,b in zip(data['predictions'],[-1,1,1,-1]))
