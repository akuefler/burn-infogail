import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from rllab import config

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix
import matplotlib
import matplotlib.patches as mpatches
import argparse

from trn.config import best_epochs
from myscripts import _create_env, _create_policy, _restore_model_args, \
nan_stack_ragged_array, gather_dicts, _create_aux_networks, \
_restore_baseline_args, _create_encoder

parser = argparse.ArgumentParser()
args = parser.parse_args()

linestyles = ["-","--","-.",":"]
markers = ["o","D","X","*","v"]
sizes = [200, 200, 200, 500, 200]

#colors = ["k","b","g","y","r","m","k"]
#colors = ["r","g","b","c"]#"r","m","k"]
colors = [
        (202./255,0.,32./255),
        (244./255,165./255,130./255),
        (146./255,197./255,222./255),
        (5/255.,113/255.,176/255.)]
#colors += colors

plt.xticks([])
plt.yticks([])

#colors = np.array([
#    [64,0,75],
#    [118,42,131],
#    [153,112,171],
#    [194,165,207],
#    [231,212,232],
#    [247,247,247],
#    [217,240,211],
#    [166,219,160],
#    [90,174,97],
#    [27,120,55],
#    [0,68,27]
#]) / 255.
exp_name = "../data/models/17-06-13/CORL2-06130728-JTZM-7"
#path = "../data/models/17-06-13/CORL2-06130728-JTZM-7/valid/expert_errors_denorm0_infoprior0_deterministic0.h5"
path ="../data/models/17-06-13/CORL2-06130728-JTZM-7/valid/expert_errors_denorm0_infoprior0_deterministic0.h5"
mpl = 300

#model_names = ["Burn-InfoGAIL", "InfoGAIL","GAIL", "VAE"] # "..."]
model_names = ["Burn-InfoGAIL", "InfoGAIL", "GAIL", "VAE"] # "..."]

font = {'family': 'normal',
        #'weight': 'bold',
        'size': 12}
matplotlib.rc('font', **font)


with h5py.File(path, "r") as hf:
    mf = hf["measure_features"][...]
    z = hf["z"][...]

if True:
    fig_name = "style_spaghetti"
    f, axs = plt.subplots(1,1,figsize=(15,10))
    axs = [axs]
    patches = []

    K = 500
    k = 9
    linewidth = 2
    counts = [0,0,0,0]
    for j in range(0,len(z)):
        if counts[z[j]] > k:
            continue
        xx = mf[j,1,:mpl]
        yy = mf[j,2,:mpl]

        sty = linestyles[z[j]]
        size = sizes[z[j]]
        mk = markers[z[j]]

        color = colors[z[j]]
        axs[0].plot(xx,yy,linewidth=linewidth,c=color,linestyle=sty,zorder=0)
        axs[0].scatter(xx[-1],yy[-1],c=color,s=size,marker=mk,zorder=10)

        axs[0].set_xticklabels([])
        axs[0].set_yticklabels([])

        axs[0].set_xticks([])
        axs[0].set_yticks([])

        counts[z[j]] += 1
    print(counts)
    assert(all([c == k + 1 for c in counts]))

    axs[0].set_xlim((-70,290))#320))
    axs[0].set_ylim((-20,120))
else:
    fig_name = "embedding"
    f, axs = plt.subplots(1,1)
    args = _restore_model_args(exp_name, args, exclude_keys =[])
    env = _create_env(args, encoder = None)
    policy, init_ops = _create_policy(args,env)
    with tf.Session() as sess:
        policy.load_params(exp_name, best_epochs.get(exp_name,-1), [])
        embedding_tensor = None
        for tensor in tf.trainable_variables():
            if tensor.name == 'policy/mean_network/z/hidden_0/W:0':
                embedding_tensor = tensor
                embed = tensor.eval()
        lda = PCA(n_components= 2)
        embed_ = lda.fit(embed).transform(embed)
        for i in range(4):
            axs.scatter(embed_[i,0], embed_[i,1],
                    c=colors[i],marker=markers[i],s=sizes[i])
        print("get params")

    D = distance_matrix(embed, embed)
    #axs.set_title("Embedding Space",fontweight="bold")
    plt.sca(axs)

save = True
if save:
    print(config.PROJECT_PATH)
    #plt.sca(axs)
    plt.savefig("{}/{}.pdf".format(config.PROJECT_PATH+"/figs", fig_name),dpi=800,format="pdf",
            bbox_inches="tight")
            #bbox_extra_artist=(lgd,))
else:
    plt.show()

