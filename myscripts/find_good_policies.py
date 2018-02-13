import argparse
import os

import numpy as np

import ast

from rllab.config import LOG_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--threshold",type=float,default=0.45)
parser.add_argument("--header",type=str,default="train_ami_cls")
parser.add_argument("--greater_than",type=int,default=1)
#parser.add_argument("--job_name",type=str,default="JBOTH05142329")
parser.add_argument("--job_name",type=str,default="CORL2")
parser.add_argument("--exclude_keys",type=str,nargs="+",default=[])
parser.add_argument("--exclude_values",type=str,nargs="+",default=[])

parser.add_argument("--only_show_h5",type=int,default=1)

args = parser.parse_args()

exclude_values = []
for ev in args.exclude_values:
    try:
        ev = ast.literal_eval(ev)
    except ValueError:
        pass
    exclude_values.append(ev)

good_models = []

def compare_to_thresh(x):
    flag = False
    if args.greater_than:
        flag = np.max(x) > args.threshold
        if flag:
            print("epoch: {} of {}".format(np.argmax(x),len(x)))
    else:
        flag =  np.min(x) < args.threshold
        if flag:
            print("epoch: {} of {}".format(np.argmin(x),len(x)))
    return flag

def read_tab(path):
    X = np.genfromtxt(path, delimiter=',')
    if X.shape == (0,):
        return
    X = X[1:]
    headers = [header.replace('"','') for header in np.genfromtxt(path, dtype=str, delimiter=',')[0]]
    if args.header not in headers:
        return
    ret = X[:,headers.index(args.header)]
    if compare_to_thresh(ret):
        return path

def read_args(path):
    with open(path,'r') as f:
        model_args = ''.join(f.readlines())
        model_args = model_args.replace("null","None")
        model_args = model_args.replace("false","False")
        model_args = model_args.replace("true","True")
        model_args = eval(model_args)
    return model_args

def recurse(path, models):
    #models = []
    d_model = []
    for subdir in os.listdir(path):
        new_path = "{}/{}".format(path,subdir)
        if os.path.isdir(new_path):
            if "tab.txt" in os.listdir(new_path):
                if args.job_name in new_path:
                    #print("Found: {}".format(new_path))
                    model_args = read_args(
                            new_path + '/args.txt'
                            )
                    excludes = [model_args[key] == val for key, val in
                            zip(args.exclude_keys,exclude_values)]
                    if not any(excludes):
                        model = read_tab(
                            new_path + '/tab.txt'
                        )
                    else:
                        model = None
                    if args.only_show_h5:
                        if not os.path.isfile("{}/epochs.h5".format(new_path)):
                            model = None
                    if model is not None:
                        models.append(model)
            else:
                d_model = recurse(new_path, models)

    models += d_model

    return list(set(models)) # so hacky ...

def main():
    log_dir = LOG_DIR
    good_models = recurse(log_dir, [])
    if good_models == []:
        print("There are no good models.")
    else:
        print("GOOD MODELS:")
        for gm in good_models:
            print(gm)

    print("There are {} good models".format(len(good_models)))


if __name__ == "__main__":
    main()

halt= True
