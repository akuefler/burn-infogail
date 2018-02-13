import numpy as np
import argparse

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import h5py

from rllab import config

import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default= "")  #e.g., "models/17-01-25/Pendulum-2"
parser.add_argument('--itrs',type=int,nargs="+",default=[])
parser.add_argument('--mode',type=str,default="gradients")
parser.add_argument('--save',type=int,default=0)

args = parser.parse_args()
assert args.mode in ["weights","encodings","gradients","loss","features","rnn_features"]

sources = []
targets = []
ex_targets = []
weights = []
grad_mags = []
losses_ex = []
losses_pi = []
losses_de = []

features_pi = []
features_ex = []

with h5py.File("../data/{}/epochs.h5".format(args.exp_name),"r") as hf:
    keys = hf.keys()
    if args.itrs == []:
        r = range(0,len(keys)-1)
    else:
        r = args.itrs

    for itr in r:
        if args.mode == "encodings":
            sources.append(hf[keys[itr]]['source'][...])
            targets.append(hf[keys[itr]]['target'][...])
            ex_targets.append(hf[keys[itr]]['ex_target'][...])
        if args.mode == "weights":
            weights.append(hf[keys[itr]]['w_target0'][...])
        if args.mode == "gradients":
            grad_mags.append(hf[keys[itr]]['grad_mags'][...]
            )
        if args.mode == "loss":
            losses_ex.append(hf[keys[itr]]['loss_ex'][...]
            )
            losses_pi.append(hf[keys[itr]]['loss_pi'][...]
            )
            losses_de.append(hf[keys[itr]]['loss_de'][...]
            )
        if args.mode in ["features","rnn_features"]:
            features_pi.append(hf[keys[itr]]['{}_pi'.format(args.mode)][...]
            )
            features_ex.append(hf[keys[itr]]['{}_ex'.format(args.mode)][...]
            )
    
#fig = plt.figure(figsize=plt.figaspect(0.5))
fig = plt.figure(figsize=(30,10))

markersize = 65
for i in range(len(args.itrs)):
    if args.mode == "encodings":
        source = sources[i]
        target = targets[i]
        ex_target = ex_targets[i]        

        ax = fig.add_subplot(1,len(args.itrs),1 + i, projection='3d')
        ax.scatter(source[:,0],source[:,1],source[:,2],c='r',s=markersize) # real datapoints
        ax.scatter(target[:,0],target[:,1],target[:,2],c='b',s=markersize) # learned encoding
        ax.scatter(ex_target[:,0],ex_target[:,1],ex_target[:,2],c='g',s=markersize) # learned encodings
        
    if args.mode == "features" or args.mode == "rnn_features":
        ax = fig.add_subplot(1,len(args.itrs),1 + i)
        F, y = [], []
        for j, feature_set in enumerate([features_ex, features_pi]):
            feat = feature_set[i]
            if args.mode == "features":
                feat = np.reshape(feat, [-1, feat.shape[-1]])
            else:
                feat = feat[:,-1,:]
                
            F.append(feat)
            y.append(j * np.ones(feat.shape[0]))

        X = np.row_stack(F)
        y = np.concatenate(y)
        
        pca = PCA(n_components=2)
        projected = pca.fit_transform(X)
        #projected = X
        
        C = np.eye(3)
        ax.scatter(projected[:,0],projected[:,1],c=np.eye(3)[y.astype('int32')])
        
if args.mode == "gradients":
    G = np.array(grad_mags)
    deviants = np.where(np.abs(G.mean(axis=0)) >= np.median(G) + np.std(G))
    
    for g in range(G.shape[-1]):
        if g in deviants:
            continue
        plt.plot(G[:,g])
    #plt.plot([g[0] for g in grad_mags_pi],color='b')
    #plt.plot([g[1] for g in grad_mags_pi],color='r')
    #plt.plot([g[2] for g in grad_mags_pi],color='r')
    #plt.plot([g[3] for g in grad_mags_pi],color='r')
    
if args.mode == "loss":
    #ax1 = fig.add_subplot(111)
    #ax2 = fig.add_subplot(121)
    #ax3 = fig.add_subplot(131)
    
    #ax1.plot(losses_ex)
    #ax2.plot(losses_pi)
    #ax3.plot(losses_de)
    f, axs = plt.subplots(1,3)
    for ax, loss in zip(axs, [losses_ex, losses_de, losses_pi]):
        ax.plot(loss)

if args.mode == "weights":
    w = [np.column_stack((weight,np.nan * np.ones(weight.shape[0]))) for weight in weights]
    W = np.concatenate(w,axis=1)
    cmap = plt.cm.bwr
    cmap.set_bad('black',1.)
    
    plt.matshow(W,cmap=cmap)
    lim = np.abs(np.nanmax(W))
    plt.clim(-lim,lim)
    
#if args.mode == "features":
    #f, axs = plt.subplots(1,2)
    #for ax, features, color in zip(axs,[features_pi, features_ex],["b","r"]):
        #if features.ndim > 2:
            #np.reshape(features, [-1, features[-1]])
            #ax.scatter(features,c=color)
            
    
#for (source, target ax) in zip(sources,targets,axs):
    #ax.scatter(source[:,0],source[:,1],source[:,2],c='r',s=markersize) # real datapoints
    #ax.scatter(target[:,0],target[:,1],target[:,2],c='b',s=markersize) # learned encodings
    
if args.save:
    plt.savefig("{}/{}.pdf".format(config.FIG_DIR,args.mode),format='pdf')

plt.show()
