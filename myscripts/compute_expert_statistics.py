import tensorflow as tf
import numpy as np

from trn.config import expert_data_paths, policy_paths
from myscripts import _create_env, _create_policy, _create_expert_data, \
    _create_aux_networks, _restore_model_args

import argparse
import h5py
import rltools.util

import matplotlib.pyplot as plt

import scipy
from scipy.stats import mode, entropy

import itertools

from sklearn.metrics import accuracy_score, mutual_info_score

from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()

#parser.add_argument("--environment",type=str,default='JNGSIM')
parser.add_argument("--discrete",type=int,default=1)

# necessary ... 
parser.add_argument("--use_infogail",type=int,default=1)
parser.add_argument("--use_valid",type=int,default=1)
parser.add_argument("--policies",type=str,nargs="+",default=[
    "models/17-06-07/EXPERIMENT4-06070746-JTZM-0"])

args = parser.parse_args()

HEADERS = ['state/desiredAng_B_T_Ds', 'state/ittc_B_T_Ds', 'state/jerk_B_T_Ds',
        'state/laneOffsetL_B_T_Ds', 'state/laneOffsetR_B_T_Ds',
        'state/speed_B_T_Ds', 'state/timeConAcc_B_T_Ds',
        'state/timeConBrake_B_T_Ds', 'state/timegap_B_T_Ds',
        'state/turnRateG_B_T_Ds']

class Model(object):
    def predict():
        pass

    @property
    def is_recurrent():
        pass

    @property
    def z_dim():
        pass

class KMeans_Model(Model):
    def __init__(self, k = 4):
        self._model = KMeans(n_clusters = k)
        self._z_dim = k

    def predict(self, obs, act):
        X = np.column_stack([obs, act])
        acc = None
        print("Fitting K means ...")
        Y_B = self._model.fit_predict(X)
        return Y_B, acc

    @property
    def z_dim(self):
        return self._z_dim

    @property
    def is_recurrent(self):
        return False

def cluster_to_label(centroids,labels,n_class= None):
    if n_class is None:
        n_class = len(np.unique(labels))
    PERM = np.row_stack(list(itertools.permutations(range(n_class))))
    accs = []
    for perm in PERM:
        c = perm[centroids]
        acc = accuracy_score(labels,c)
        accs.append(acc)
        #accs.append(accuracy_score(c, labels))
    i = np.argmax(accs)
    return PERM[i][centroids], accs[i]

def _plot_gauss_means(ax, expert_data):
    obs_B_T_Do = expert_data["obs"]
    act_B_T_Da = expert_data["act"]

    z_mean = expert_data["z_mean"]

    speed_B_T = obs_B_T_Do[:,:,2]
    speed_B = speed_B_T.mean(axis=1)

    ax.scatter(z_mean[:,0],z_mean[:,1],c=speed_B, s= 30)
    ax.set_title(name)

def empiricalKLD(metrics, cls_B, y_B, n_bins = 100, smoothing = 1):
    """
    a list of numpy arrays
    """
    chi_d = lambda x, y : np.sum( ((x - y) ** 2) / (y).astype('float32') )

    n_classes = len(np.unique(cls_B))
    n_metrics = len(metrics)
    K = np.zeros((n_metrics, n_classes)).astype('float32')
    CHI = np.zeros((n_metrics, n_classes)).astype('float32')
    for i, metric in enumerate(metrics):
        for cls in range(n_classes):
            x_cls = metric[cls_B == cls]
            x_y = metric[y_B == cls]
            h_cls, edges = np.histogram(x_cls, bins = n_bins)
            h_y, _ = np.histogram(x_y, bins = edges)

            h_y += smoothing
            h_cls += smoothing

            K[i, cls] = entropy(h_cls, h_y)
            CHI[i, cls] = chi_d(h_y, h_cls)

    return K, CHI

def _label_metrics(args, data, model = None):
    info = {}

    obs_B_T_Do = data['obs']
    act_B_T_Da = data['act']
    len_B = data['exlen_B']
    len_B = np.minimum(len_B, args.max_path_length)
    cls_B = data.get('cls_B',None)
    info['cls_B'] = cls_B

    B, T, Do = obs_B_T_Do.shape
    B, T, Da = act_B_T_Da.shape

    # truncate trajectories
    T = args.max_path_length
    obs_B_T_Do = obs_B_T_Do[:,:T,:]
    act_B_T_Da = act_B_T_Da[:,:T,:]
    if model is not None:
        if not model.is_recurrent:
            obs_BT_Do = np.reshape(obs_B_T_Do, (B * T, Do))
            act_BT_Da = np.reshape(act_B_T_Da, (B * T, Da))

            obs_BT_Do = obs_BT_Do[~np.isnan(obs_BT_Do).any(axis=1)]
            act_BT_Da = act_BT_Da[~np.isnan(act_BT_Da).any(axis=1)]

            y_BT, _ = model.predict(np.concatenate([obs_BT_Do,
                np.zeros_like(obs_BT_Do)[:,:model.z_dim]], axis=-1),
                act_BT_Da)
            y_B_T = np.reshape(y_BT, (B,T))
            y_B, _ = np.squeeze(mode(y_B_T,axis=1))
        if model.is_recurrent:
            y_B, _ = model.predict(np.concatenate([obs_B_T_Do,
                np.zeros_like(obs_B_T_Do)[:,:,:model.z_dim]], axis=-1),
                act_B_T_Da, EOS = len_B)
    else:
        y_B = data["z_mean"].argmax(axis=1)

        y_B = cls_B

    # compute emergent metrics
    if 'met' in data.keys():
        states = data["met"]
    else:
        states, _ = _create_state_matrix(data)

    speed = np.nanmean(states[:,:,HEADERS.index('state/speed_B_T_Ds')],axis=-1)
    ittc = np.nanmean(states[:,:,HEADERS.index('state/timegap_B_T_Ds')],axis=-1)
    turnRateG = np.nanmean(np.abs(states[:,:,HEADERS.index('state/turnRateG_B_T_Ds')]),axis=-1)

    metrics = [speed, ittc] #, turnRateG]

    # if labels exist, "convert" clusters to labels
    if cls_B is not None:
        info['MI'] = mutual_info_score(y_B, cls_B)
        y_B, acc = cluster_to_label(y_B,cls_B,n_class= None)
        info['ACC'] = acc
        _KLD, _CHI = empiricalKLD(metrics, y_B, cls_B)
        info["KLD"] = np.sum(_KLD)
        info["CHI"] = np.sum(_CHI)

    return y_B, metrics, info

def _plot_discrete(args, axvec, policy_data, expert_data, model = None,
        measures = [], baselines = []):

    # compute label
    colors = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]])
    print("loading expert labels")
    y_B_ex_Q, metrics_ex, info_Q = _label_metrics(args, expert_data, model)

    print("loading policy labels")
    y_B_pi, metrics_pi, _ = _label_metrics(args, policy_data)

    bdict = {"KM":KMeans_Model(k=model.z_dim)}
    mdict = {"ACC": " Accuracy: {} \n",
             "MI" :" Mut. Inf: {} \n",
             "KLD":" KL Diver: {} \n",
             "CHI":" Chi Sqrd: {} \n"}

    y_B_ex_Bs, info_Bs = [], []
#    for baseline in baselines:
#        y_B_ex_B, _, info_B = _label_metrics(args, expert_data, KMeans_Model(k=model.z_dim))
#        y_B_ex_Bs.append(y_B_ex_B)
#        info_Bs.append(info_B)

    caption = "".join([mdict[m] for m in measures])
    if 'ACC' in info_Q.keys():
        #axvec[0].text(0.1,0.8,"Accuracy: {}".format(info_Q['acc']),fontsize=12)
        new_cap = caption.format(*[info_Q[m] for m in measures])
        axvec[1].set_xlabel(new_cap)

    for i, info_B in enumerate(info_Bs):
        if 'acc' in info_B.keys():
            axvec[2 + i].set_xlabel(caption.format(*[info_B[m] for m in measures]))

    #y_Bs =[y_B_ex_Q, y_B_ex_K, y_B_pi]
    #M = [metrics_ex, metrics_ex, metrics_pi]
    y_Bs = [y_B_ex_Q] + y_B_ex_Bs
    M = [metrics_ex] * len(y_Bs)

    if info_Q['cls_B'] is not None:
        y_Bs.insert(0, info_Q['cls_B'])
        M.insert(0, metrics_ex)
    else:
        y_Bs = [None] + y_Bs
        M = [None] + M

    for k, (y_B, metrics) in enumerate(zip(y_Bs,M)):
        if y_B is None:
            continue
        plot_to_ax(axvec[k], y_B, metrics)

def plot_to_ax(ax, y_B, metrics, normalize = True):
    y = [[] for i in range(len(metrics))]
    for i in np.unique(y_B):
        for j, metric in enumerate(metrics):
            y[j].append(metric[y_B == i].mean())

    x = np.array(range(len(y[0])))
    colors = ['r','g','b','y']
    for i, y_ in enumerate(y):
        if normalize:
            y_ = np.array(y_) / np.sum(y_)
        _ = ax.bar(x + (0.2 * i), y_, width=0.2, color=colors[i])

def _create_state_matrix(expert_data, normalize=True):
    keys = sorted(filter(lambda x : "state/" in x, expert_data.keys()))
    X = []
    for key in keys:
        X.append(expert_data[key])
    if X != []:
        X = np.concatenate(X, axis=-1)

    return X, keys

if __name__ == "__main__":
    #baselines = ["KM"]
    baselines = []
    measures = ["MI","KLD"]
    f, axs = plt.subplots(len(args.policies), 2 + len(baselines))
    if axs.ndim == 1:
        axs = axs[None,...]

    axs[0,0].set_title("True Label, Expert Trajs")
    axs[0,1].set_title("Q-Label, Expert Trajs")
    for i in range(1,1 + len(baselines)):
        axs[0, i].set_title("{}, Expert Trajs".format(baselines[i - 1]))

    for i, path in enumerate(args.policies):
        print(path)
        axrow = axs[i]
        tf.reset_default_graph()

        exp_name = '../data/{}'.format(path)
        args = _restore_model_args(exp_name, args)
        args.trajdata_indices = [1]
        env = _create_env(args)
        _, _, info_model, env = _create_aux_networks(args, env)

        expert_data_T, expert_data_V = _create_expert_data(args)
        policy_data_T, policy_data_V = _create_expert_data(args, path=path)
        if args.discrete:
            with tf.Session() as sess:
                info_model.load_params(exp_name, -1, [])
                _plot_discrete(args, axrow, policy_data_V, expert_data_V, model =
                        info_model, baselines = baselines, measures = measures)
        else:
            _plot_gauss_means(ax, expert_data)
        axrow[0].set_ylabel(path.split("/")[-1])

    # axes must have same limits
    y_lo = min([ax.get_ylim()[0] for axrow in axs for ax in axrow])
    y_hi = max([ax.get_ylim()[1] for axrow in axs for ax in axrow])
    for axrow in axs:
        for ax in axrow:
            ax.set_ylim(y_lo, y_hi)

    plt.show()

    import pdb; pdb.set_trace()

