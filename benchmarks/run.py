import sys,time,statistics,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scalargrad.features import *
durations=[]
for _ in range(5):
    start=time.perf_counter()
    fit(MLP([2,4,1]),[[-1,-1],[-1,1],[1,-1],[1,1]],[-1,1,1,-1],50)
    durations.append(time.perf_counter()-start)
print(json.dumps({'workload':'fit(MLP([2,4,1]),[[-1,-1],[-1,1],[1,-1],[1,1]],[-1,1,1,-1],50)','repetitions':len(durations),'median_seconds':statistics.median(durations),'min_seconds':min(durations)}))
