import argparse,json
from .features import MLP,fit
def main():
    parser=argparse.ArgumentParser(description='Train a scalar-autodiff MLP on XOR')
    parser.add_argument('--steps',type=int,default=1000);parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    try:
        x=[[-1,-1],[-1,1],[1,-1],[1,1]];y=[-1,1,1,-1]
        model=MLP([2,4,1],args.seed);losses=fit(model,x,y,args.steps)
        print(json.dumps({'initial_loss':losses[0],'final_loss':losses[-1],'predictions':[model(row)[0].data for row in x],'parameters':len(model.parameters())}))
    except (ValueError,TypeError) as error:parser.error(str(error))
