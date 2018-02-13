import h5py
import matplotlib.pyplot as plt
import numpy as np

from rllab import config

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

font = {'family': 'normal',
        #'weight': 'bold',
        'size': 18}
matplotlib.rc('font', **font)

#colors = ["k","b","g","y","r","m","k"]
colors = ["b","g","y","m"]#"r","m","k"]
linestyles = ["-","--","-.",":"]
markers = ["*","D","X","o","v"]
sizes = [900, 200, 200, 200, 200]
sizes_ = [25, 15, 20, 20, 20]
colors += colors
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
MEASURES = [("measure/speed", "Speed (m/s)"),
            ("measure/pos", "Position (m)")]
            #("measure/posFt", "Lane Offset (m)")]

"""
python sample_policy.py --exp_name models/17-06-13/CORL2-06130728-JTZM-7 --denorm 0 --use_info_prior 1 --deterministic 1 --batch_size 1000; python sample_policy.py --exp_name models/17-06-13/CORL2-06130728-JTZM-4 --denorm 0 --use_info_prior 0 --deterministic 1 --batch_size 1000; python sample_policy.py --exp_name models/17-06-17/CORL3-06172024-JTZM-0 --denorm 0 --use_info_prior 0 --deterministic 1 --batch_size 1000; python sample_policy.py --exp_name vae --denorm 1 --use_info_prior 1 --deterministic 1 --batch_size 1000;
"""
paths = [
        # ours:
        "../data/models/17-06-13/CORL2-06130728-JTZM-7/valid/expert_errors_denorm0_infoprior1_deterministic1.h5",
        #"../data/models/17-06-13/CORL2-06130728-JTZM-7/valid/expert_errors_denorm0_infoprior0_deterministic1.h5",
        # standard infogail
        "../data/models/17-06-13/CORL2-06130728-JTZM-4/valid/expert_errors_denorm0_infoprior0_deterministic1.h5",
        # standard gail
        "../data/models/17-06-17/CORL3-06172024-JTZM-0/valid/expert_errors_denorm0_infoprior0_deterministic1.h5",
        # recurrent version
        #"../data/models/17-06-19/CORL4-06182344-JTZM-1/valid/expert_errors_denorm0_infoprior1_deterministic0.h5"
        # something else
        #"../data/models/17-06-13/CORL2-06130728-JTZM-8"
        # VAE
        "../data/baselines/vae/valid/expert_errors_denorm1_infoprior1_deterministic1.h5",
        ]

model_names = ["Burn-InfoGAIL", "InfoGAIL","GAIL", "VAE"] # "..."]
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off

ERRORS = []
RMSEs = []
CLSs = []
M = []
MEASURE_FEATS = []
TRUE_FEATS = []
for path in paths:
    with h5py.File(path, "r") as hf:
        RMSE = hf["RMSE"][...]
        mf = hf["measure_features"][...]
        tf = hf["true_features"][...]
        measures = list(hf["measures"][...])
        try:
            classes = np.array(hf["classes"][...]) - 1.
        except:
            classes = None
        errors = np.array(hf["errors"][...])

    RMSEs.append(RMSE)
    M.append(measures)
    CLSs.append(classes)
    ERRORS.append(errors)
    MEASURE_FEATS.append(mf)
    TRUE_FEATS.append(tf)

if True:
    figsize = [3 * 4, 4 * 4]
    f, axs = plt.subplots(len(MEASURES),1, figsize=figsize)#(4 * 4,1 * 4))
    patches = []
    for i, (measure, ylabel) in enumerate(MEASURES):
        for j, RMSE in enumerate(RMSEs):
    #        if k < 4 and CLSs[j] is not None:
    #            cls = np.array(CLSs[j])
    #            err = ERRORS[j]
    #            RMSE = (err[cls == k]).mean(axis=(0,1))
            m_ix = M[j].index(measure)
            axs[i].plot(RMSE[m_ix], color=colors[j], linestyle=linestyles[j], linewidth=3)

            if i == 0:
                patches.append(mlines.Line2D([],[],color=colors[j], label=
                    model_names[j],linestyle=linestyles[j],linewidth=3))

        x_tick_labels = np.arange(-5,35,5)
        axs[i].set_xticklabels(x_tick_labels)
        axs[i].set_xlabel("Horizon (s)", fontweight="bold")
        axs[i].set_ylabel(ylabel, fontweight="bold")

    name = "rmse"
else:
    #s = 32 #looks good
    s = 5
    #s = 27
    K = 2
    f, axs = plt.subplots(1,1,figsize=(15,10))
    axs = [axs]
    patches = []
    for i, mf in enumerate([TRUE_FEATS[0]] + MEASURE_FEATS):
        if i > 0:
            color = colors[i - 1]
            label = model_names[i - 1]
            linewidth = 5
        else:
            color = 'k'
            label = 'Truth'
            linewidth = 5

        #for j in range(s,s+K):
        xx = mf[s,1,:]
        yy = mf[s,2,:]

        #xx_ = truth['measure/pos_x']
        #yy_ = truth['measure/pos_y']

        axs[0].plot(xx,yy,linewidth=linewidth,c=color,zorder=0,linestyle=linestyles[i-1])
        #plt.plot(xx_,yy_,c='k')
        axs[0].scatter(xx[-1],yy[-1],marker=markers[i-1],s=sizes[i-1],c=color, zorder=10)
        #plt.scatter(xx_[-1],yy_[-1],marker="*",s=65,c='k')

        #patches.append(mpatches.Patch(color=color,label=label))
        patches.append(mlines.Line2D([],[],markersize=sizes_[i-1],marker=markers[i-1],linewidth=0,c=color,label=label))
        axs[0].set_xticklabels([])
        axs[0].set_yticklabels([])

        axs[0].set_xticks([])
        axs[0].set_yticks([])

        axs[0].set_xlim((-70,290))#320))
        axs[0].set_ylim((-20,120))

    name = "model_spaghetti"

assert len(model_names) == len(RMSEs)
lgd = axs[0].legend(loc='upper left', handles = patches, ncol = 1)# len(model_names))

save = True
if save:
    print(config.PROJECT_PATH)
    plt.savefig("{}/{}.pdf".format(config.PROJECT_PATH+"/figs",name),dpi=800,format="pdf",
            bbox_inches="tight")
            #bbox_extra_artist=(lgd,))
else:
    plt.show()

