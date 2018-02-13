import numpy as np
import matplotlib.pyplot as plt

import os
import ast
import argparse
from collections import OrderedDict, Iterable

import pickle
import joblib

from rllab import config
from trn.config import expert_data_paths

import matplotlib.patches as mpatches

parser = argparse.ArgumentParser()
parser.add_argument('--exp_names',type=str,nargs='+')
#parser.add_argument('--job_name',type=str,default="JBOTH05132349")
#parser.add_argument('--job_name',type=str,default="JBOTH05142329")
parser.add_argument('--job_name',type=str,default="EXPERIMENT1-06021215")
#JDEBUG05181521-JTZM-BCINIT0-4
#parser.add_argument('--job_name',type=str,default="JNGSIM05101843")
# JTZ05011719-JTZ-BCINIT0-0
parser.add_argument('--iters',type=int,nargs='+')
parser.add_argument('--max_iter',type=int,default=-1)
parser.add_argument('--include_xtick',type=int,default=1)
parser.add_argument('--save',type=int,default=0)
#parser.add_argument('--include_ylabel',type=int,default=1)
parser.add_argument('--show_labels',type=int,default=0)
parser.add_argument('--y_lo',type=int,default=-1)
parser.add_argument('--y_hi',type=int,default=-1)

parser.add_argument('--ax_ix',type=int,nargs="+",default=[-1])
parser.add_argument('--color_ix',type=int,nargs="+",default=[-1])

parser.add_argument('--show_invdyn_per',type=int,default=1)
parser.add_argument('--show_gan_acc',type=int,default=1)
parser.add_argument('--show_aux_gan_acc',type=int,default=0)
parser.add_argument('--show_path_length',type=int,default=1)

parser.add_argument('--axis_arg',type=str,nargs="+",default='')
parser.add_argument('--color_arg',type=str,nargs="+",default='')

parser.add_argument('--exclude_keys',type=str,nargs="+",default=[])
parser.add_argument('--exclude_values',type=str,nargs="+",default=[])

#parser.add_argument('--highlight_ix',type=int,default=-1)
#parser.add_argument("--highlight_name",type=str,default="JTZ05081943-JTZM-BCINIT0-8")
#parser.add_argument("--highlight_name",type=str,default="JTZ05081943-JTZM-BCINIT0-54")

parser.add_argument("--color_spectrum",type=int,default=0)

parser.add_argument("--highlight_name",type=str,default="")

args = parser.parse_args()

max_iters = args.max_iter

assert len(args.exclude_keys) == len(args.exclude_values)

exclude_values = []
for ev in args.exclude_values:
    try:
        ev = ast.literal_eval(ev)
    except ValueError:
        pass
    exclude_values.append(ev)


def gather_experiments(job_name):
    S = []
    for direct in os.listdir(config.LOG_DIR):
        for model in os.listdir("{}/{}".format(config.LOG_DIR,direct)):
            if all([kw in model.split("-") for kw in job_name.split("-")]):
                S.append("{}/{}/{}".format(config.LOG_DIR,direct,model))
    return [str(SS) for SS in S]

if args.exp_names is None:
    assert args.job_name is not None
    exp_names = gather_experiments(args.job_name)
else:
    exp_names = args.exp_names

if args.iters is None:
    iters = [-1] * len(exp_names)
else:
    iters = args.iters

def add_to_X_and_L(KEY, X, L, M, exp_name, header, headers, n_itr):
    if header in headers:
        x = M[:,headers.index(header)] # headers.index(header)
        print("{} ... {}".format(len(x),n_itr))
        if len(x) == n_itr:
            try:
                X[header][KEY].append(x)
            except:
                X[header][KEY] = [x]
            try:
                L[header][KEY].append(exp_name)
            except:
                L[header][KEY] = [exp_name]
    return X, L

def construct_ix_list(arg):
    arg = tuple(arg)
    axis_arg_values = []
    D = {}
    i = 0
    for exp_name in exp_names:
        with open('{}/args.txt'.format(exp_name),'r') as f:
            model_args = ''.join(f.readlines())
            model_args = model_args.replace("null","None")
            model_args = model_args.replace("false","False")
            model_args = model_args.replace("true","True")
            model_args = eval(model_args)

        if any([model_args.get(key, None) == value for key, value in zip(args.exclude_keys,
            exclude_values)]):
            continue
        tup = []
        for rg in arg:
            mrg = model_args[rg]
            if isinstance(mrg,Iterable):
                mrg = tuple(mrg)

            if not ((rg in args.exclude_keys) and (str(mrg) in args.exclude_values)):
                tup.append(mrg)

        if len(tup) > 0:
            tup = tuple(tup)
            axis_arg_values.append(tup)

        i += 1

    unique_values = list(set(axis_arg_values))
    ax_ixs = [unique_values.index(tup) for tup in axis_arg_values]
    D = {str(zip(arg,tup)) : v for v, tup in enumerate(unique_values)}

    return ax_ixs, D

# plotting
if args.axis_arg != "":
    ax_ixs, axis_D = construct_ix_list(args.axis_arg)

else:
    if args.ax_ix == [-1]:
        ax_ixs = range(0,len(exp_names))
    else:
        assert len(args.ax_ix)
        ax_ixs = args.ax_ix
if args.color_arg != "":
    color_ixs, color_D = construct_ix_list(args.color_arg)
else:
    color_ixs = args.color_ix

headers_of_interest = []
headers_of_interest = ["pathLengths", "info_acc"] #"info_loss"]
#        ["pathLengths", "info_acc", "info_loss"] + ["AverageEnvReturn","AverageDiscountedReturn"]
#healders_of_interest += \
#["info_loss", "ave_reward"]
headers_of_interest += \
        ["train_ami_cls"]
#headers_of_interest += \
#        ["valid_ami_cls", "valid_ami_dom"]
#headers_of_interest += ["valid_ami_cls"]
#headers_of_interest += \
#        ["valid_ami_cls_in_dom_0.0", "valid_ami_cls_in_dom_1.0", "valid_ami_cls_in_dom_2.0",
#        "valid_ami_cls_in_dom_3.0"]
#headers_of_interest += \
#        ["sample_freq_0", "sample_freq_1", "sample_freq_2", "sample_freq_3"]
#headers_of_interest += \
#        ["nmi_cls_in_dom_0.0", "nmi_cls_in_dom_1.0", "nmi_cls_in_dom_2.0",
#        "nmi_cls_in_dom_3.0"]

#f, axs = plt.subplots(nrows= np.max(ax_ixs) + 1, ncols= len(headers_of_interest), figsize=(20,10))
f, axs = plt.subplots(nrows= np.max(ax_ixs) + 1, ncols=
        len(headers_of_interest), figsize=(20,5))

if axs.ndim == 1:
    axs = axs[None,...]

#colors = np.array([[1,0,0],
#                   [0,0.7,0],
#                   [0,0,1],
#                   [0.8,0.8,0],
#                   [1,0,1],
#                   [0,1,1]])
colors = np.array([[1.,0.,0.],
                    [0.,0.7,0.],
                    [0.,0.,1.],
                    [1.,0,1.],
                    [0.7,0.7,0.],
                    [0.,0.7,0.7],
                    [0.5,0.5,0.5],
                    [0.1, 0.5, 0.9]])
#ax_color_ix = np.zeros((np.max(ax_ixs) + 1, len(colors))).astype('int32')
patches = {}
#X_aveReturn = {}
#X_ganAccs = {}
#X_invdynLoss = {}
#X_invdynPred = {}
X = OrderedDict({})
L = OrderedDict({})
for header in headers_of_interest:
    X[header] = {} # stores data
    L[header] = {} # stores experiment names

print(axis_D.keys())
i = 0
for exp_name in exp_names:

    # Pull arguments for this experiment
    if not os.path.exists('{}/args.txt'.format(exp_name)):
        print("WARNING, SKIPPING: {}".format(exp_name))
        continue

    with open('{}/tab.txt'.format(exp_name),'r') as f:
        headers = f.readline().split(',')

    with open('{}/args.txt'.format(exp_name),'r') as f:
        model_args = ''.join(f.readlines())
        model_args = model_args.replace("null","None")
        model_args = model_args.replace("false","False")
        model_args = model_args.replace("true","True")
        model_args = eval(model_args)

    if any([model_args.get(key, None) == value for key, value in zip(args.exclude_keys,
        exclude_values)]):
        continue

    axrow = axs[ax_ixs[i]]

    c_ix = model_args.get('color_ix',color_ixs[i])
    KEY = (ax_ixs[i],c_ix)

    M = np.genfromtxt('{}/tab.txt'.format(exp_name), delimiter=',')[1:]
    try:
        headers = [header.replace('"','') for header in
                np.genfromtxt('{}/tab.txt'.format(exp_name), dtype=str, delimiter=',')[0]]
    except IndexError:
        print("Skipping!")
        continue

    try:
        expert_M = \
            np.genfromtxt('{}/data/{}/tab.txt'.format(config.PROJECT_PATH,expert_data_paths[model_args["environment"]]), delimiter=',')[1:]
        expert_headers = [header.replace('"','') for header in
                np.genfromtxt('{}/data/{}/tab.txt'.format(config.PROJECT_PATH,
                    expert_data_paths[model_args["environment"]]), dtype=str,
                    delimiter=',')[0]]
        maxExpEnvReturn = expert_M[:max_iters,expert_headers.index('AverageReturn')].max()
        print(maxExpEnvReturn)
        axrow[0].plot(np.ones_like(averageEnvReturn) * maxExpEnvReturn, 'b')
    except:
        pass

#    color = colors[c_ix]
#    if color == 'g':
#        halt = True
    try:
        averageEnvReturn = M[:max_iters,headers.index('AverageEnvReturn')]
    except ValueError:
        averageEnvReturn = M[:max_iters,headers.index('AverageReturn')]

    print(exp_name)
    for header in headers_of_interest:
        n = max_iters
        if n == -1:
            n = model_args["n_itr"] - 1

        X, L = add_to_X_and_L(KEY, X, L, M[:max_iters], exp_name, header, headers, n)

    if args.show_labels:
        #axrow[0].set_ylabel(exp_name.split('/')[-1], labelpad= (i % 2) * 20)
        axrow[0].set_ylabel(str(zip(args.axis_arg,[model_args[rg] for rg in
            args.axis_arg])), labelpad=(ax_ixs[i] % 2) * 30, fontweight="bold")

    i += 1

patches = [mpatches.Patch(color=colors[val], label=name) for name, val in color_D.items()]
axrow = axs[-1]
ax = axrow[len(axrow)/2]
#ax.legend(loc='lower center', handles = patches, ncol = len(patches), bbox_to_anchor=(0.5,-0.4))
ncols = np.min([len(patches),3])
ax.legend(loc='lower center', handles = patches, ncol = ncols,
        bbox_to_anchor=(0.0,-0.3))

for i, (title, X_D) in enumerate(X.items()):
    for key in X_D.keys():
        X = np.row_stack(X_D[key])
        ax_key = key[0]
        co_key = key[1]

        spectr = np.linspace(0.,1.,len(X))
        spectb = np.linspace(1.,0.,len(X))
        for j, x in enumerate(X):
            if args.color_spectrum:
                color = np.array([spectr[j],0.,spectb[j]])
                axs[ax_key][i].plot(x,color=color,linewidth=2)
            else:
                axs[ax_key][i].plot(x,color=colors[co_key] + 0.7 * (1.-colors[co_key]),zorder=0)
            #if j == args.highlight_ix:
            if args.highlight_name == L[title][key][j].split("/")[-1]:
                axs[ax_key][i].plot(x,color=colors[co_key], zorder = 1, linewidth = 2)
        if args.highlight_name == '' or args.highlight_name is None:
            axs[ax_key][i].plot(X.mean(axis=0),color=colors[co_key],linewidth=2,zorder=1)
        if args.show_labels:
            axs[ax_key][i].set_title(title)
        if title == "disc_racc":
            axs[ax_key][i].set_ylim((0.,1.))
        if "sample_freq" in title:
            axs[ax_key][i].set_ylim((0.,1.))
        if title == "valid_ami_cls" or title == "valid_ami_dom":
            axs[ax_key][i].set_ylim((0.0,0.6))

# set limits
for axrow in axs:
    ylo, yhi = axrow[0].get_ylim()
    if args.y_lo != -1:
        axrow[0].set_ylim(args.y_lo, yhi)

    ylo, yhi = axrow[0].get_ylim()
    if args.y_hi != -1:
        axrow[0].set_ylim(ylo, args.y_hi)

if args.save:
    print(config.PROJECT_PATH)
    plt.savefig("{}/epochs.pdf".format(config.PROJECT_PATH+"/figs"),dpi=800,format="pdf")
else:
    plt.show()

